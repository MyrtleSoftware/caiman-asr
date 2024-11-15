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

from caiman_asr_train.data.external_source.core import str_to_numpy_unicode
from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import (
    IDENTITY_NORMALIZE_CONFIG,
    NormalizeConfig,
)


class LengthUnknownError(Exception):
    """
    The length of a webdataset is unknown.
    """

    pass


class WebDatasetReader:
    """
    This class reads samples of data from tar-file webdatasets.

    This format enables reading from a tar/zip file shard containing a collection of samples
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
    https://github.com/webdataset/webdataset/issues/237. As a workaround, replace the
    `.` with `_` of filename except the extension part.

    For more information on the format there is a good description in this third-party
    library: https://webdataset.github.io/webdataset/ but note that
    webdataset library is not used here. Instead, torchdata's webdataset reading
    functionality is used.

    Parameters
    ----------
    tokenizer
        Optional tokenizer to use to tokenize the transcripts. If None, then the string
        transcripts are returned. If None, then charset must be passed.
    shuffle
        Whether to shuffle the data.
    file_root
        The root directory for the tar/zip file paths. This is passed to the root argument
        of the torchdata FileLister class.
    tar_files
        List of tar/zip files to read from. This is passed to the masks argument
        of the torchdata FileLister class.
        There are two modes for this argument:
        1) (If file_root is passed): this should be a list of filenames (or fileglobs)
        within the file_root directory. NOTE: in this mode, all of your tar/zip files must
        be in a single flat directory.
        2) (If file_root is falsy or == "/"): this should be a list of absolute paths
        or globs of the tar/zip files.
        3) The files parsed must be in either tar or zip format, and not a mix of both.
    charset
        Optional List of strings containing the supported characters. This is passed as
        an alternative to Tokenizer when the transcripts are not tokenized.
    normalize_config
        Config that controls transcript normalization
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
    min_duration
        The minimum duration of a sample in seconds. If a sample is shorter than this it
        is filtered out.
    """

    audio_suffixes = {".flac", ".wav"}

    def __init__(
        self,
        tokenizer: Optional[Tokenizer],
        shuffle: bool,
        file_root: Optional[str],
        tar_files: List[str],
        charset: Optional[List[str]] = None,
        normalize_config: NormalizeConfig = IDENTITY_NORMALIZE_CONFIG,
        shuffle_buffer_size: int = 20000,
        max_duration: float = float("inf"),
        max_transcript_len: float = float("inf"),
        num_buckets: int = 1,
        batch_size: Optional[int] = None,
        bucketing_buffer_size: int = 4,
        skip_audio: bool = False,
        sample_rate: int = 16000,
        min_duration: int | float = 0.05,
    ) -> None:
        assert tar_files, "must specify tar_files "

        if tokenizer is None:
            assert charset is not None, (
                "charset must be passed if tokenizer is None in order to perform "
                "normalization"
            )

        self.tar_files = tar_files
        self.file_root = file_root
        self.charset = charset
        self.tokenizer = tokenizer
        self.normalize_config = normalize_config
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_duration = max_duration
        self.max_transcript_len = max_transcript_len
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.bucketing_buffer_size = bucketing_buffer_size
        self.skip_audio = skip_audio
        self.sample_rate = int(sample_rate)
        self.min_duration = min_duration

        if not file_root or file_root == "/":
            # self.tar_files is one or more absolute paths/globs
            file_lister = FileLister(self.tar_files, recursive=True)
        else:
            # self.tar_files must be one or more filenames/file globs within file_root
            # Note that not setting recursive=True here, means that self.tar_files must be
            # at the top level in the self.file_root directory
            file_lister = FileLister(self.file_root, self.tar_files)

        if self.shuffle:
            # shuffle twice: once at the tar file level and once at the sample level
            # this first shuffle is so that each node in a distributed setting reads
            # different tar files each epoch
            file_lister = file_lister.shuffle()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_tar_files = len(list(file_lister))
        assert num_tar_files >= world_size, (
            f"Number of tar/zip files ({num_tar_files}) must be greater than or "
            f"equal to the number of nodes ({world_size}) otherwise data "
            "cannot be sharded across nodes"
        )
        # It is required that num_workers * world_size be at most the number of tar files
        # so that each worker has a tar file to read. A max number of 4 workers
        # per process can be used.
        num_workers = min(num_tar_files // world_size, 4)
        # apply sharding across nodes here
        file_lister = file_lister.sharding_filter()

        self.opener = FileOpener(file_lister, mode="b")

        loaded_format = self.file_opener_load_format()

        self._webdataset_pipe = loaded_format.map(self._decode).webdataset()

        self._webdataset_pipe = self._webdataset_pipe.filter(self._filter_fn)
        self._webdataset_pipe = self._webdataset_pipe.map(self._norm_and_tokenize)
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

    def file_opener_load_format(self):
        """
        Load the tar/zip files in the correct format.
        """
        if self.tar_files[0].endswith(".zip"):
            return self.opener.load_from_zip()
        if self.tar_files[0].endswith(".tar"):
            return self.opener.load_from_tar()
        raise ValueError("Only zip and tar files are supported")

    @staticmethod
    def _manipulate_key(key):
        """
        This will replace the periods in the last part of the key except the file extension.
        Example:
        >>> key = '/datasets/XYZdata.tar/jobid1234.wav_aligned.attributeABC.<ext>'
        >>> WebDatasetReader._manipulate_key(key)
        '/datasets/XYZdata.tar/jobid1234_wav_aligned_attributeABC.<ext>'
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
            if length_secs > self.max_duration or length_secs < self.min_duration:
                data = None
            return key, data
        if key.endswith(".txt"):
            transcript = value.read().decode("utf-8")
            if len(transcript) > self.max_transcript_len:
                return key, None
            return key, transcript
        else:
            raise ValueError(f"Unknown file type: {key}")

    def _norm_and_tokenize(self, item):
        raw_transcript = item[".txt"]
        norm_transcript = norm_and_tokenize(
            transcript=raw_transcript,
            tokenizer=self.tokenizer,
            normalize_config=self.normalize_config,
            charset=self.charset,
        )
        item[".txt"] = (
            norm_transcript
            if self.tokenizer is None
            else np.array(norm_transcript, dtype=np.int32)
        )
        item["raw_transcript_array"] = str_to_numpy_unicode(raw_transcript)
        return item

    def _filter_fn(self, item):
        """
        Filter out any samples that don't have both a transcript and audio.

        This should only occur if the transcript or audio was explicitly set to None in
        the _decode function.
        """
        try:
            transcript_not_none = item[".txt"] is not None
        except KeyError as exc:
            raise ValueError(
                f"There isn't a paired text and audio file for {item=}\n"
                "At tarfile creation time, make sure that each audio file is stored "
                "sequentially with its .txt transcript. You can check the file order "
                "for a given tar file in a bash shell with `$ tar tf <tarfile path>.tar"
            ) from exc
        return transcript_not_none and (
            self.skip_audio or self._get_audio(item) is not None
        )

    def _bucketing(self, pipe):
        """
        Apply bucketing to the samples so similar length samples appear in the same batch.

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

        # Unbatch this and dali will re-batch it w/o a shuffle maintaining
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

        Use of the self._webdataset_pipe will do most of the work here.
        next(self._iter) returns a dictionary with keys corresponding to the file
        suffixes. For example, in this (audio dataset) case the dictionary looks like this:
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
            assert audio_item is not None, (
                f"No audio files found for sample={key} in dict with "
                f"keys={sample_dict.keys()}"
            )

        transcripts = sample_dict[".txt"]
        return (
            audio_item,
            transcripts,
            sample_dict["raw_transcript_array"],
            str_to_numpy_unicode(key),
        )
