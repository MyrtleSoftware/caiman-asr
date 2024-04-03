#!/usr/bin/env python3
from argparse import ArgumentParser
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Optional


@beartype
def add_basic_hugging_face_args(parser: ArgumentParser) -> None:
    hf = parser.add_argument_group("Hugging Face datasets")
    hf.add_argument(
        "--use_hugging_face",
        "--use_hf",
        action="store_true",
        help="Use a Hugging Face dataset for validation",
    )
    hf.add_argument(
        "--hugging_face_val_dataset",
        "--hf_val_dataset",
        type=str,
        help="The name of the Hugging Face dataset to validate on",
    )
    hf.add_argument(
        "--hugging_face_val_config",
        "--hf_val_config",
        type=str,
        default=None,
        help="""When loading the val dataset from the Hugging Face Hub,
        this option allows you to specify the configuration if needed.
        Defaults to None""",
    )
    hf.add_argument(
        "--hugging_face_val_split",
        "--hf_val_split",
        type=str,
        default="validation",
        help="""The split to use for validation. Defaults to 'validation'.
        This supports the TensorFlow Slicing API, e.g. 'validation[25%:75%]'
        will validate on the middle two quarters of the validation set.
        See https://www.tensorflow.org/datasets/splits for more
        """,
    )
    hf.add_argument(
        "--hugging_face_val_transcript_key",
        "--hf_val_transcript_key",
        type=str,
        default="text",
        help="""The key in the Hugging Face dataset that refers to the transcript.
        Defaults to "text", but could be something else, like "transcription".
        To find out the key, please look at the dataset's documentation
        on the Hugging Face Hub""",
    )


@beartype
@dataclass
class HuggingFaceArgs:
    dataset: str
    split: str
    transcript_key: str
    config: Optional[str]


@beartype
def build_hugging_face_args(
    use_hugging_face: bool,
    dataset: Optional[str],
    split: str,
    transcript_key: str,
    config: Optional[str],
) -> Optional[HuggingFaceArgs]:
    if use_hugging_face:
        return HuggingFaceArgs(
            dataset=dataset,
            split=split,
            transcript_key=transcript_key,
            config=config,
        )
    else:
        assert dataset is None, "Did you forget to set --use_hugging_face?"
        return None
