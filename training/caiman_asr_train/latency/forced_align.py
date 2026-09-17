import logging
import math
import os
import re
import string
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torchaudio
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from lhotse import CutSet, SupervisionSegment
from lhotse.cut import Cut
from lhotse.supervision import AlignmentItem
from torchaudio import functional as F
from torchaudio.functional import TokenSpan
from torchaudio.transforms import Resample

from caiman_asr_train.data.text.allowed_charset import (
    get_allowed_charset,
    iso_unidecode,
)
from caiman_asr_train.latency.ctm import get_abs_manifest_paths
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.utils.iter import flat

# disable nemo's internal logging (OneLogger and Telemetry) that
# won't be suppressed any other way
logging.disable(logging.CRITICAL)
import nemo.collections.asr as nemo_asr  # noqa


def parse_args() -> Namespace:
    parser = ArgumentParser("Perform forced alignment and write CTM files.")
    parser.add_argument(
        "--dataset_dir", required=True, type=str, help="Root dir of dataset"
    )
    parser.add_argument(
        "--manifests",
        required=False,
        type=str,
        nargs="+",
        help="Paths of the dataset manifest files - either absolute or relative "
        "to dataset_dir."
        "Ignored if --read_from_tar=True",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Optional directory to output .ctm files. "
            "Use --overwrite to use existing directory",
        ),
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        help="ISO lang code of the transcripts to be aligned",
    )
    parser.add_argument(
        "--model_name", type=str, help="Nemo model to use for alignment"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Use CPU instead of GPU."
    )
    parser.add_argument(
        "--segment_len",
        type=int,
        default=5,
        help="""Audio is split into segments of this length in minutes. Forced
        alignment is run on each segment and the results are then concatenated.""",
    )

    return parser.parse_args()


@beartype
@dataclass
class AlignmentModel:
    model: nemo_asr.models.ASRModel
    blank_id: int
    unk_id: int
    stride: float
    sample_rate: int
    unsupported_punc: str
    device: str


@beartype
def _load_model(model_name: str, device: str) -> AlignmentModel:
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Cannot load model {model_name}: {e}")

    model.to(device).eval()
    model.cur_decoder = "ctc"
    blank_id = model.ctc_decoder.num_classes_with_blank - 1
    unk_id = model.ctc_decoder.vocabulary.index("<unk>")
    stride = model.cfg.encoder.subsampling_factor * model.cfg.preprocessor.window_stride
    sample_rate = model.cfg.preprocessor.sample_rate
    unsupported_punc = "".join(
        [punc for punc in string.punctuation if punc not in model.tokenizer.vocab]
    )
    return AlignmentModel(
        model=model,
        blank_id=blank_id,
        unk_id=unk_id,
        stride=stride,
        sample_rate=sample_rate,
        unsupported_punc=unsupported_punc,
        device=device,
    )


@beartype
def _clean_unsupported_chars(
    text: str, charset: List[str], unsupported_punc: str
) -> str:
    text = iso_unidecode(text, charset)
    text = re.sub(rf"[{re.escape(unsupported_punc)}]", " ", text)
    return text.strip()


@beartype
def _word_to_ids(
    text: str, tokenizer, charset: List[str], unsupported_punc: str
) -> List[Dict[str, str]]:
    word_to_id_map = []

    for word in text.split():
        clean_word = _clean_unsupported_chars(word, charset, unsupported_punc)
        if clean_word:
            ids = tokenizer.text_to_ids(clean_word)

            word_to_id_map.append(
                {"original_word": word, "token_ids": ids, "num_tokens": len(ids)}
            )

    return word_to_id_map


@beartype
def _load_audio(audio_file: str, model_sample_rate: int) -> torch.Tensor:
    """Loads an audio file into a waveform and resamples if necessary."""
    waveform, sample_rate = torchaudio.load(audio_file)

    # Check if audio is stereo (2 channels)
    if waveform.shape[0] == 2:
        # Keep only the first channel (left channel)
        waveform = waveform[0].unsqueeze(0)

    if sample_rate != model_sample_rate:
        print_once(
            f"WARNING: Resampling audio from {sample_rate} to {model_sample_rate}"
        )
        resample = Resample(sample_rate, model_sample_rate)
        # bundle sample rate is 16k
        waveform = resample(waveform)

    return waveform


@beartype
class ForcedAligner:
    def __init__(
        self,
        lang_code: str,
        segment_length: int,
        device: str,
        model_name: Optional[str] = None,
    ):
        self.lang_code = lang_code

        if model_name is None:
            model_name = f"nvidia/stt_{lang_code}_fastconformer_hybrid_large_pc"
        self.aligner = _load_model(model_name, device)
        self.normalization_charset = get_allowed_charset([self.lang_code])
        self.segment_length = segment_length * 60

    def align(
        self, audio_path: str, transcript: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, str]]]:
        """Align audios with their ground truth transcripts"""

        aligned_tokens = torch.Tensor([])
        alignment_scores = torch.Tensor([])
        with torch.no_grad():
            audio_wav = _load_audio(audio_path, self.aligner.sample_rate)
            segment_length = self.segment_length * self.aligner.sample_rate
            num_samples = audio_wav.shape[1]
            outputs = []
            start = 0

            while start < num_samples:
                end = min(start + segment_length, num_samples)
                # Pass each waveform segment to model
                segment = audio_wav[:, start:end]
                hypotheses = self.aligner.model.transcribe(
                    segment.squeeze(0),
                    return_hypotheses=True,
                    verbose=False,
                )
                _log_probs = (
                    hypotheses[0].y_sequence.unsqueeze(0).to(self.aligner.device)
                )
                outputs.append(_log_probs)
                start = end

            # Concatenate model outputs for each segment
            log_probs = torch.cat(outputs, dim=1)

            word_to_ids_map = _word_to_ids(
                transcript,
                self.aligner.model.tokenizer,
                self.normalization_charset,
                self.aligner.unsupported_punc,
            )
            target_ids = flat(word_map["token_ids"] for word_map in word_to_ids_map)
            target_ids_tensor = torch.tensor([target_ids]).to(self.aligner.device)
            try:
                aligned_tokens, alignment_scores = F.forced_align(
                    log_probs,
                    target_ids_tensor,
                    blank=self.aligner.blank_id,
                )
            except Exception as e:
                print(f"WARNING: Alignment failed for Transcript={transcript}: {e}")

            return aligned_tokens, alignment_scores, word_to_ids_map

    def process_alignment(
        self,
        aligned_tokens: torch.Tensor,
        alignment_scores: torch.Tensor,
        word_to_ids_map: List[Dict[str, str]],
    ) -> List[AlignmentItem]:
        token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])
        token_spans = list(
            filter(
                lambda token_span: token_span.token != self.aligner.blank_id,
                token_spans,
            )
        )

        words = []
        if token_spans:
            offset = 0
            stride = self.aligner.stride
            for word_map in word_to_ids_map:
                lens = word_map["num_tokens"]
                if lens == 1 and word_map["token_ids"][0] == self.aligner.unk_id:
                    continue

                word_span = token_spans[offset : offset + lens]
                start = round(word_span[0].start * stride, 3)
                end = word_span[-1].end * stride
                score = round(_score(word_span), 2)

                words.append(
                    AlignmentItem(
                        symbol=word_map["original_word"],
                        start=start,
                        duration=round(end - start, 3),
                        score=score,
                    )
                )
                offset += lens

        return words


@beartype
def get_output_filepath(
    data_filepath: str, output_dir: str, overwrite: bool = False
) -> str:
    """Creates output filename based on the data file's base name.

    Ensures file doesn't already exist in the output directory.
    """

    base_name = Path(data_filepath).name
    output_file = os.path.join(output_dir, base_name)

    # Check if the file already exists
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"The file {output_file} already exists." "Use --overwrite to overwrite"
        )

    return output_file


@beartype
def _contains_digits(text: str) -> bool:
    if re.findall(r"[0-9]", text):
        return True
    return False


@beartype
def _align(
    audio_path: str, transcript: str, forced_aligner: ForcedAligner
) -> List[AlignmentItem]:
    # Don't generate alignments if transcripts contain digits
    if _contains_digits(transcript) or len(transcript.split()) < 3:
        return []

    alignments, alignment_scores, word_to_ids_map = forced_aligner.align(
        audio_path, transcript
    )

    if alignments.nelement() == 0:
        return []

    word_alignments = forced_aligner.process_alignment(
        alignments, alignment_scores, word_to_ids_map
    )

    return word_alignments


@beartype
def _process_supervision(
    sup: SupervisionSegment, audio_path: str, forced_aligner: ForcedAligner
) -> SupervisionSegment:
    sup = sup.with_alignment(
        kind="word",
        alignment=_align(
            audio_path=audio_path, transcript=sup.text, forced_aligner=forced_aligner
        ),
    )
    return sup


@beartype
def _process_cut(cut: Cut, forced_aligner: ForcedAligner, data_dir: str) -> Cut:
    audio_path = f"{data_dir}/{cut.recording.sources[0].source}"
    cut = cut.map_supervisions(
        partial(
            _process_supervision, audio_path=audio_path, forced_aligner=forced_aligner
        )
    )
    return cut


@beartype
def _score(spans: List[TokenSpan]) -> float:
    return math.exp(sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans))


@beartype
def process_manifest_file(
    manifest_path: str, output_file: str, data_dir: str, forced_aligner: ForcedAligner
):
    """
    Prepares manifests for alignment and saves the generated alignments to file

    Args:
        manifest_path (str): The path to the dataset manifest file.
        output_file (str): The path to the output manifest file with alignments
        data_dir (str): The directory where audio files referenced in the manifest are
            stored.
        forced_aligner (ForcedAligner): The model to be used for forced alignment
    """

    print(f"Aligning manifest: {Path(manifest_path).name}")

    cutset = CutSet.from_jsonl(manifest_path)

    # NB: we deliberately do NOT resample the cutset here. `align()` loads the audio
    # straight from disk via `_load_audio`, which resamples to the model's rate
    # internally, so the alignment is unaffected by the cut's recording metadata.
    # Resampling the cutset would rewrite the recording sample_rate/num_samples/
    # duration (and add a Resample transform) in the emitted manifest, leaving the
    # output differing from the input in more than just the added alignments.

    cutset = cutset.map(
        partial(_process_cut, forced_aligner=forced_aligner, data_dir=data_dir)
    )

    CutSet(cutset).to_file(output_file)


def main(args: Namespace):
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda"

    if Path(args.output_dir).exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory {args.output_dir} already exists."
            "Use --overwrite to overwrite existing files"
        )
    assert (
        args.output_dir != args.dataset_dir
    ), "Output directory cannot be the same as the dataset directory"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    assert (args.lang_code is None) ^ (
        args.model_name is None
    ), "Only one of lang_code or model_name can be provided"
    forced_aligner = ForcedAligner(
        lang_code=args.lang_code,
        segment_length=args.segment_len,
        model_name=args.model_name,
        device=device,
    )

    manifests = get_abs_manifest_paths(args.dataset_dir, args.manifests)

    for manifest in manifests:
        output_file = get_output_filepath(
            manifest, args.output_dir, overwrite=args.overwrite
        )
        process_manifest_file(manifest, output_file, args.dataset_dir, forced_aligner)


if __name__ == "__main__":
    args = parse_args()
    main(args)
