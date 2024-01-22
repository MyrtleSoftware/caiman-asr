import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch.distributed as dist
from beartype.typing import List, Optional
from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    MultiProcessingReadingService,
    SequentialReadingService,
)
from torchdata.datapipes.iter import BucketBatcher, FileLister, FileOpener

from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.text.preprocess import norm_and_tokenize


class LengthUnknownError(Exception):
    """
    The length of a webdataset is unknown.
    """

    pass


class WebDatasetReader:
    """
    This class reads samples of data from tar-file webdatasets.

    This format enables reading from a tar file shard containing a collection of samples
    with the convention that all files associated with a single sample have the same key.
    For example, for a simple audio dataset shard with just two samples might include
    the following four files:
        webdatasetExample/
        ├── clip1.flac
        ├── clip1.txt
        ├── clip2.flac
        └── clip2.txt
    where the flac files are the audio samples and the text files are the corresponding
    transcripts.
    NOTE: if filenames contain more than one period then webdataset considers filename
    until first period as the key and remaining part as extension. It is generally not
    recommended to use multiple periods in a filename as mentioned here
    https://github.com/webdataset/webdataset/issues/237. As a workaround we replace the
    `.` with `_` of filename except the extension part.

    For more information on the format there is a good description in this third-party
    library: https://webdataset.github.io/webdataset/ but note that we don't use this
    webdataset library here, instead we use the torchdata's webdataset reading
    functionality.

    Parameters
    ----------
    tokenizer
        Optional tokenizer to use to tokenize the transcripts. If None, then the string
        transcripts are returned. If None, then charset must be passed.
    shuffle
        Whether to shuffle the data.
    file_root
        The root directory for the tar file paths. This is passed to the root argument
        of the torchdata FileLister class.
    tar_files
        List of tar files to read from. This is passed to the masks argument
        of the torchdata FileLister class.
        There are two modes for this argument:
        1) (If file_root is passed): this should be a list of filenames (or fileglobs)
        within the file_root directory. NOTE: in this mode, all of your tar files must
        be in a single flat directory.
        2) (If file_root is falsy or == "/"): this should be a list of absolute paths
        or globs of the tar files.
    charset
        Optional List of strings containing the supported characters. This is passed as
        an alternative to Tokenizer when the transcripts are not tokenized.
    normalize_transcripts
        Whether to normalize the transcripts.
    shuffle_buffer_size
        The size of the sample shuffle buffer. This must be larger than the number of
        samples in a shard so that cross-shard shuffling is performed. The larger this
        value, the larger the memory footprint.
    max_duration
        The maximum duration of a sample in seconds. If a sample is longer than this it
        is filtered out.
    max_transcript_len
        The maximum length of a transcript in characters. If a transcript is longer than
        this it is filtered out.
    num_buckets
        The number of buckets to use for bucketing samples by duration.
    batch_size
        The batch size to use. Note that batching is not performed in this WebDatasetReader
        class: hence the batch_size is only required if using bucketing (i.e. if
        num_buckets > 1).
    bucketing_buffer_size
        The number of batches that are considered in the pool when bucketing. Increasing
        this parameter will give more accurate and stable bucketing, but will increase
        memory consumption.
    skip_audio
        Whether to skip reading the audio files. This is useful to save time doing
        I/O and audio decoding if you just want to view the transcripts.
    sample_rate
        The sample rate of the audio files.
    """

    audio_suffixes = {".flac", ".wav"}

    def __init__(
        self,
        tokenizer: Optional[Tokenizer],
        shuffle: bool,
        file_root: Optional[str],
        tar_files: List[str],
        charset: Optional[List[str]] = None,
        normalize_transcripts: bool = False,
        shuffle_buffer_size: int = 20000,
        max_duration=float("inf"),
        max_transcript_len=float("inf"),
        num_buckets: int = 1,
        batch_size: Optional[int] = None,
        bucketing_buffer_size: int = 4,
        skip_audio: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        assert tar_files, "must specify tar_files "

        if tokenizer is None and normalize_transcripts:
            assert (
                charset is not None
            ), "charset must be passed if tokenizer is None in order to perform normalization"

        self.tar_files = tar_files
        self.file_root = file_root
        self.charset = charset
        self.tokenizer = tokenizer
        self.normalize_transcripts = normalize_transcripts
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_duration = max_duration
        self.max_transcript_len = max_transcript_len
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.bucketing_buffer_size = bucketing_buffer_size
        self.skip_audio = skip_audio
        self.sample_rate = int(sample_rate)

        if not file_root or file_root == "/":
            # self.tar_files is one or more absolute paths/globs
            file_lister = FileLister(self.tar_files, recursive=True)
        else:
            # self.tar_files must be one or more filenames/file globs within file_root
            # Note that we don't set recursive=True here so all self.tar_files must be
            # at the top level in the self.file_root directory
            file_lister = FileLister(self.file_root, self.tar_files)

        if self.shuffle:
            # we shuffle twice: once at the tar file level and once at the sample level
            # this first shuffle is so that each node in a distributed setting reads
            # different tar files each epoch
            file_lister = file_lister.shuffle()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_tar_files = len(list(file_lister))
        assert num_tar_files >= world_size, (
            f"Number of tar files ({num_tar_files}) must be greater than or "
            f"equal to the number of nodes ({world_size}) otherwise we can't shard data "
            "across nodes"
        )
        # We need num_workers * world_size to be at most the number of tar files
        # so that each worker has a tar file to read. We also want as many
        # workers per process as possible, up to a maximum of 4.
        num_workers = min(num_tar_files // world_size, 4)
        # apply sharding across nodes here
        file_lister = file_lister.sharding_filter()

        self.opener = FileOpener(file_lister, mode="b")

        self._webdataset_pipe = (
            self.opener.load_from_tar().map(self._decode).webdataset()
        )

        self._webdataset_pipe = self._webdataset_pipe.filter(self._filter_fn)
        if self.shuffle:
            # shuffle the samples
            self._webdataset_pipe = self._webdataset_pipe.shuffle(
                buffer_size=self.shuffle_buffer_size
            )
        self._webdataset_pipe = self._bucketing(self._webdataset_pipe)

        # finally, create the reading service and dataloader
        if world_size == 1:
            rs = None
        else:
            mp_rs = MultiProcessingReadingService(num_workers=num_workers)
            dist_rs = DistributedReadingService()
            rs = SequentialReadingService(dist_rs, mp_rs)
        self.dataloader = DataLoader2(self._webdataset_pipe, reading_service=rs)

    @staticmethod
    def _manipulate_key(key):
        """
        This will replace the periods in the last part of the key except the file extension.
        E.g., The key `/datasets/XYZdata.tar/jobid1234.wav_aligned.attributeABC.<ext>` is changed to
        `/datasets/XYZdata.tar/jobid1234_wav_aligned_attributeABC.<ext>`.
        """
        key_path = Path(key)
        new_key = str(
            (key_path.parent / key_path.stem.replace(".", "_")).with_suffix(
                key_path.suffix
            )
        )
        return new_key

    def _decode(self, item):
        """
        Apply decoding to the webdataset item.
        """
        key, value = item
        key = self._manipulate_key(key)
        if any(key.endswith(audio_suff) for audio_suff in self.audio_suffixes):
            if self.skip_audio:
                return key, None
            audio_bytes = value.read()
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

            length_secs = len(data) / self.sample_rate
            if length_secs > self.max_duration:
                data = None
            return key, data
        if key.endswith(".txt"):
            transcript = value.read().decode("utf-8")
            if len(transcript) > self.max_transcript_len:
                return key, None
            transcript = norm_and_tokenize(
                transcript,
                self.tokenizer,
                self.normalize_transcripts,
                self.charset,
            )
            if self.tokenizer is not None:
                transcript = np.array(transcript, dtype=np.int32)
            return key, transcript
        else:
            raise ValueError(f"Unknown file type: {key}")

    def _filter_fn(self, item):
        """
        Filter out any samples that don't have both a transcript and audio.

        This should only occur if the transcript or audio was explicitly set to None in
        the _decode function.
        """
        try:
            transcript_not_none = item[".txt"] is not None
        except KeyError:
            raise ValueError(
                f"There isn't a paired text and audio file for {item=}\n"
                "At tarfile creation time, make sure that each audio file is stored "
                "sequentially with its .txt transcript. You can check the file order "
                "for a given tar file in a bash shell with `$ tar tf <tarfile path>.tar"
            )
        return transcript_not_none and (
            self.skip_audio or self._get_audio(item) is not None
        )

    def _bucketing(self, pipe):
        """
        Apply bucketing to the samples so that similar length samples appear in the same batch.

        See BucketingSampler for more details.
        """
        if self.num_buckets <= 1:
            return pipe
        assert self.shuffle, "Can't run bucketing without shuffling"
        assert self.batch_size is not None, "Must specify batch_size for bucketing"
        bucketed_pipe = BucketBatcher(
            pipe,
            self.batch_size,
            batch_num=self.num_buckets,
            bucket_num=self.bucketing_buffer_size,
            use_in_batch_shuffle=False,
            sort_key=self._sort_bucket_fn,
        )

        # we then unbatch this and dali will re-batch for us w/o a shuffle maintaining
        # the batching of the samples
        return bucketed_pipe.unbatch()

    def _sort_bucket_fn(self, bucket):
        return sorted(bucket, key=lambda item: len(self._get_audio(item)))

    def _get_audio(self, item):
        audio_item = None
        for suffix in self.audio_suffixes:
            audio_item = item.get(suffix)
            if audio_item is not None:
                break
        return audio_item

    def __iter__(self):
        self._iter = iter(self.dataloader)
        return self

    def __next__(self):
        """
        Return sample of data from the webdataset.

        This returns a sample of audio data and its corresponding tokenized transcripts.

        We use the self._webdataset_pipe to do most of the work here.
        next(self._iter) returns a dictionary with keys corresponding to the file
        suffixes. In our (audio dataset) case we have something like:
        {
            ".txt": <transcript>,
            ".flac": <audio samples np array>,
            "__key__": <sample id>
        }
        """
        sample_dict = next(self._iter)
        key = sample_dict["__key__"]
        audio_item = self._get_audio(sample_dict)

        if not self.skip_audio:
            assert (
                audio_item is not None
            ), f"No audio files found for sample={key} in dict with keys={sample_dict.keys()}"

        transcripts = sample_dict[".txt"]
        return audio_item, transcripts
