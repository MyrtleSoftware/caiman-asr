#!/usr/bin/env python3
# Use forced alignment to obtain timestamps from audio data
# https://pytorch.org/audio/master/tutorials/ctc_forced_alignment_api_tutorial.html

import argparse
import os
import tarfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import yaml
from beartype import beartype
from beartype.typing import Generator, List, Tuple, Union
from torchaudio.functional import TokenSpan
from torchaudio.transforms import Resample
from tqdm import tqdm

from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.latency.ctm import get_abs_manifest_paths, get_abs_tar_paths
from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
    normalize_config_from_full_yaml,
)
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.utils.fast_json import fast_read_json


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser("Perform forced alignment and write CTM files.")
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
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    parser.add_argument(
        "--tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the dataset tar files." "Ignored if --read_from_tar=False.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Optional directory to output .ctm files. Defaults to --dataset_dir.",
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
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Only reads the allowed characters from the model config file",
    )

    return parser.parse_args()


@beartype
@dataclass
class AlignmentData:
    waveform: torch.Tensor
    transcript: List[str]
    audio_filepath: str


@beartype
def align(
    emission: torch.Tensor, tokens: List[int], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align model emissions and ground truth tokens."""
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    # remove batch dimension
    alignments, scores = (alignments[0], scores[0])
    # convert back to probability
    scores = scores.exp()
    return alignments, scores


@beartype
def unflatten(token_list: List[TokenSpan], lengths: List[int]) -> List[List[TokenSpan]]:
    """Obtain word level alignments."""
    assert len(token_list) == sum(lengths)
    i = 0
    result = []

    for length in lengths:
        result.append(token_list[i : i + length])
        i += length

    return result


@beartype
def _score(spans: List[TokenSpan]) -> float:
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


@beartype
def write_ctm(
    output_file: str,
    filename: str,
    spans: List[List[TokenSpan]],
    ratio: float,
    transcript: List[str],
    sample_rate: int,
) -> None:
    """Process forced alignment timestamps and write to ctm file."""
    with open(output_file, "a") as f:
        for i, span in enumerate(spans):
            start = int(ratio * span[0].start) / sample_rate
            end = int(ratio * span[-1].end) / sample_rate
            duration = end - start
            score = _score(span)
            f.write(
                f"{filename} 1 {start:.3f} {duration:.3f} {transcript[i]} {score:.2f}\n"
            )


@beartype
def load_audio(
    speech_file: Union[str, tarfile.ExFileObject], bundle_sample_rate: int
) -> torch.Tensor:
    """Loads an audio file into a waveform and resamples if necessary."""
    waveform, sample_rate = torchaudio.load(speech_file)

    # Check if audio is stereo (2 channels)
    if waveform.shape[0] == 2:
        # Keep only the first channel (left channel)
        waveform = waveform[0].unsqueeze(0)

    if sample_rate != bundle_sample_rate:
        print_once(
            f"WARNING: Resampling audio from {sample_rate} to {bundle_sample_rate}"
        )
        resample = Resample(sample_rate, bundle_sample_rate)
        # bundle sample rate is 16k
        waveform = resample(waveform)

    return waveform


@beartype
def process_tar_file(
    tar_path: str, bundle_sample_rate: int, device: str
) -> Generator[AlignmentData]:
    """
    Processes and prepares audio & transcript pairs from tar file for alignment processing.

    Args:
        tar_path (str): The path to the tar file containing audio and transcript files.
        bundle_sample_rate (int): The sample rate expected by the model for audio
            processing.
        device (str): CUDA or CPU.

    Returns:
        Generator[AlignmentData]: A generator of AlignmentData objects, containing
        waveform tensor, transcript, and path to audio.
    """
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        audio_files = [m for m in members if m.name.endswith((".flac", ".wav"))]

        for audio_member in audio_files:
            # Extract the corresponding transcript file
            transcript_member_name = audio_member.name.rsplit(".", 1)[0] + ".txt"
            transcript_member = tar.getmember(transcript_member_name)

            with tar.extractfile(audio_member) as speech_file, tar.extractfile(
                transcript_member
            ) as transcript_file:
                transcript = transcript_file.read().decode().strip().split()
                waveform = load_audio(speech_file, bundle_sample_rate).to(device)
                yield AlignmentData(waveform, transcript, audio_member.name)


@beartype
def count_files_in_tar(tar_path: str) -> int:
    """
    Counts the number of audio files in a tar file.

    Args:
    tar_path (str): The path to the tar file.

    Returns:
    int: The number of audio files in the tar file.
    """
    audio_extensions = {".flac", ".wav"}
    audio_count = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if any(member.name.endswith(ext) for ext in audio_extensions):
                audio_count += 1

    return audio_count


@beartype
def process_manifest_file(
    manifest_path: str, data_dir: str, bundle_sample_rate: int, device: str
) -> Generator[AlignmentData]:
    """
    Loads audio data from files given in manifest and prepares for alignment processing.

    Args:
        manifest_path (str): The path to the dataset manifest file.
        data_dir (str): The directory where audio files referenced in the manifest are
            stored.
        bundle_sample_rate (int): The sample rate expected by the model.
        device (str): CUDA or CPU.

    Returns:
        Generator[AlignmentData]: A generator of AlignmentData objects, containing
        waveform tensor, transcript, and path to audio.
    """
    data = fast_read_json(manifest_path)

    for sample in data:
        transcript = sample["transcript"].split()  # Words
        speech_file = sample["files"][0]["fname"]
        speech_filepath = os.path.join(data_dir, speech_file)
        waveform = load_audio(speech_filepath, bundle_sample_rate).to(device)
        yield AlignmentData(waveform, transcript, speech_file)


@beartype
def count_files_in_manifest(manifest_path: str) -> int:
    """
    Counts the number of files listed in a manifest file.

    Args:
    manifest_path (str): The path to the manifest file.

    Returns:
    int: The number of files listed in the manifest.
    """

    return len(fast_read_json(manifest_path))


@beartype
def get_output_filepath(data_filepath: str, output_dir: str) -> str:
    """Creates output CTM filename based on the data file's base name.

    Ensures file doesn't already exist in the output directory.
    """
    # CTM filename shares same base name as manifest/tar file
    base_name = Path(data_filepath).stem
    output_file = os.path.join(output_dir, f"{base_name}.ctm")

    # Check if the file already exists
    if os.path.exists(output_file):
        raise FileExistsError(f"The file {output_file} already exists.")

    return output_file


@beartype
def split_and_process_waveform(
    waveform: torch.Tensor,
    model: torchaudio.pipelines._wav2vec2.utils._Wav2Vec2Model,
    segment_length: int,
) -> torch.Tensor:
    """
    Splits the waveform into segments and processes each segment individually.

    Args:
    waveform (Tensor): The input waveform tensor.
    model: The wav2vec model.
    segment_length (int): Segment length in minutes.

    Returns:
    Concatenated tensor of model outputs.
    """
    segment_length *= 16000 * 60  # convert to num samples
    num_samples = waveform.shape[1]
    outputs = []
    start = 0

    with torch.inference_mode():
        while start < num_samples:
            end = min(start + segment_length, num_samples)
            # Pass each waveform segment to model
            segment = waveform[:, start:end]
            emission, _ = model(segment)
            outputs.append(emission)
            start = end
    # Concatenate model outputs for each segment

    return torch.cat(outputs, dim=1)


@beartype
def normalise(
    transcript: List[str],
    normalize_config: NormalizeConfig,
    charset: List[str],
):
    raw_transcript = " ".join(transcript)
    raw_transcript = norm_and_tokenize(raw_transcript, normalize_config, None, charset)
    raw_transcript = raw_transcript.split()

    return raw_transcript


@beartype
def perform_forced_alignment(
    data_generator: Generator[AlignmentData],
    num_samples: int,
    model: torchaudio.pipelines._wav2vec2.utils._Wav2Vec2Model,
    tokenizer: dict,
    output_file: str,
    bundle_sample_rate: int,
    device: str,
    normalize_config: NormalizeConfig,
    charset: List[str],
    segment_length: int = 5,
) -> None:
    """
    Performs forced alignment using a specified model, and writes CTM output.

    Args:
        data_generator (Generator[AlignmentData]): A generator containing waveforms,
            transcripts, and audio filepaths.
        num_samples (int): Number of samples in data_generator.
        model (torchaudio.pipelines._wav2vec2.utils._Wav2Vec2Model): The model used for
            obtaining emissions.
        tokenizer (dict): A dictionary for tokenizing the transcript.
        output_file (str): The filepath where the CTM output will be written.
        bundle_sample_rate (int): The sample rate expected by the model.
        device (str): CUDA or CPU.
    """

    pbar = tqdm(total=num_samples)
    # Iterate over utterances
    for sample in data_generator:
        # Obtain model emissions
        emission = split_and_process_waveform(sample.waveform, model, segment_length)

        normalized_transcript = normalise(sample.transcript, normalize_config, charset)

        tokenized_transcript = [
            tokenizer[c] for word in normalized_transcript for c in word
        ]

        try:
            aligned_tokens, alignment_scores = align(
                emission, tokenized_transcript, device=device
            )
        except Exception as err:
            msg = [
                "WARNING: align failed:",
                f"\t{normalized_transcript=}",
                f"\t{sample.transcript=}",
                f"\t{err=}",
            ]
            print("\n".join(msg))

            continue

        # remove repeated and blank tokens
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
        word_spans = unflatten(
            token_spans, [len(word) for word in normalized_transcript]
        )

        num_frames = emission.size(1)
        ratio = sample.waveform.size(1) / num_frames

        write_ctm(
            output_file,
            sample.audio_filepath,
            word_spans,
            ratio,
            normalized_transcript,
            sample_rate=bundle_sample_rate,
        )
        del emission, sample
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()


def main(args: Namespace):
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda"
    # Load Wave2Vec model
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device)
    tokenizer = bundle.get_dict(star=None)

    data_dir = args.dataset_dir
    # By default, export CTM files to dataset directory

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
    else:
        output_dir = data_dir

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
        normalize_config = normalize_config_from_full_yaml(model_config)
        allowed_chars = model_config["tokenizer"]["labels"]

    if args.read_from_tar:
        tar_paths = get_abs_tar_paths(data_dir, args.tar_files)

        for tar_path in tar_paths:
            output_file = get_output_filepath(tar_path, output_dir)
            data_generator = process_tar_file(
                tar_path, bundle.sample_rate, device=device
            )
            num_samples = count_files_in_tar(tar_path)
            perform_forced_alignment(
                data_generator,
                num_samples,
                model,
                tokenizer,
                output_file,
                bundle.sample_rate,
                device=device,
                segment_length=args.segment_len,
                charset=allowed_chars,
                normalize_config=normalize_config,
            )
    else:
        manifests = get_abs_manifest_paths(data_dir, args.manifests)

        for manifest in manifests:
            output_file = get_output_filepath(manifest, output_dir)
            data_generator = process_manifest_file(
                manifest, data_dir, bundle.sample_rate, device=device
            )
            num_samples = count_files_in_manifest(manifest)
            perform_forced_alignment(
                data_generator,
                num_samples,
                model,
                tokenizer,
                output_file,
                bundle.sample_rate,
                device=device,
                segment_length=args.segment_len,
                charset=allowed_chars,
                normalize_config=normalize_config,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
