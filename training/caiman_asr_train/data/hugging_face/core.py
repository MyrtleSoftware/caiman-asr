#!/usr/bin/env python3
from functools import partial

import numpy as np
from beartype import beartype
from datasets import Audio, load_dataset
from datasets.distributed import split_dataset_by_node

from caiman_asr_train.args.hugging_face import HuggingFaceArgs
from caiman_asr_train.data.external_source.core import str_to_numpy_unicode
from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


@beartype
class HuggingFaceReader:
    def __init__(
        self,
        hugging_face_args: HuggingFaceArgs,
        num_shards: int,
        shard_id: int,
        sample_rate: int,
        tokenizer: Tokenizer,
        normalize_config: NormalizeConfig,
        max_duration: int | float,
        max_transcript_length: int | float,
        min_duration: int | float = 0.05,
    ):
        dataset = load_dataset(
            hugging_face_args.dataset,
            name=hugging_face_args.config,
            split=hugging_face_args.split,
        )
        check_transcript_key(
            dataset, hugging_face_args.transcript_key, hugging_face_args.dataset
        )
        # Do maps/filters on the fly to save space:
        iterable_dataset = dataset.to_iterable_dataset(num_shards=num_shards)
        sharded_dataset = split_dataset_by_node(
            iterable_dataset, world_size=num_shards, rank=shard_id
        )
        resampled_dataset = sharded_dataset.cast_column(
            "audio", Audio(sampling_rate=sample_rate, mono=True, decode=True)
        )
        renamed_dataset = resampled_dataset.rename_column(
            hugging_face_args.transcript_key, "transcript"
        )
        filter_fn = partial(
            is_utterance_short,
            max_duration,
            sample_rate,
            max_transcript_length,
            min_duration=min_duration,
        )
        filtered_dataset = renamed_dataset.filter(filter_fn)
        map_fn = partial(tokenize_transcript, tokenizer, normalize_config)
        self.tokenized_dataset = filtered_dataset.map(map_fn).map(make_float_single)

    def __iter__(self):
        self._hf_iterator = iter(self.tokenized_dataset)
        return self

    def __next__(self):
        example = next(self._hf_iterator)
        return (
            example["audio"]["array"],
            example["tokenized_transcript"],
            example["raw_transcript_array"],
            str_to_numpy_unicode(get_fname(example)),
        )


@beartype
def is_utterance_short(
    max_duration: int | float,
    sample_rate: int,
    max_transcript_length: int | float,
    example: dict,
    min_duration: int | float = 0.05,
) -> bool:
    return (
        len(example["transcript"]) < max_transcript_length
        and len(example["audio"]["array"]) < max_duration * sample_rate
        and len(example["audio"]["array"]) > min_duration * sample_rate
    )


@beartype
def tokenize_transcript(
    tokenizer: Tokenizer,
    normalize_config: NormalizeConfig,
    example: dict,
) -> dict:
    tokenized_transcript = np.array(
        norm_and_tokenize(
            transcript=example["transcript"],
            tokenizer=tokenizer,
            normalize_config=normalize_config,
        ),
        dtype=np.int32,
    )
    example["raw_transcript_array"] = str_to_numpy_unicode(example["transcript"])
    del example["transcript"]
    example["tokenized_transcript"] = tokenized_transcript
    return example


@beartype
def make_float_single(example: dict) -> dict:
    example["audio"]["array"] = example["audio"]["array"].astype(np.float32)
    return example


@beartype
def check_transcript_key(dataset, transcript_key: str, dataset_name: str) -> None:
    if transcript_key not in dataset.column_names:
        raise ValueError(
            f"Cannot load transcripts because {dataset_name} does not "
            f"have a column named '{transcript_key}'. "
            f"Try setting the transcript key to one of {dataset.column_names}"
        )


@beartype
def get_fname(example: dict) -> str:
    # This could be expanded to check for more keys,
    # or to take a user-configurable key
    if "id" in example:
        return example["id"]
    if "audio_id" in example:
        return example["audio_id"]
    return "No ID found in example"
