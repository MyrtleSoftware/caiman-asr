# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from beartype import beartype
from beartype.typing import Optional
from nvidia.dali.plugin.numba.experimental import NumbaFunction
from nvidia.dali.types import DALIDataType
from scipy.io.wavfile import write

from caiman_asr_train.args.noise_augmentation import NoiseAugmentationArgs
from caiman_asr_train.data.cache import persistent_cache
from caiman_asr_train.data.dali.hugging_face_utils import make_noise_dataset_for_dali
from caiman_asr_train.data.dali.impulse import ImpulseAug
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.noise import (
    NoiseAugmentationIterator,
    babble_batch_dali_api,
    blend_dali_api,
)
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.hugging_face.core import HuggingFaceReader
from caiman_asr_train.data.webdataset import WebDatasetReader
from caiman_asr_train.setup.dali import DaliYAMLConfig
from caiman_asr_train.setup.text_normalization import NormalizeLevel
from caiman_asr_train.train_utils.distributed import time_print_once


class PipelineParams:
    def __init__(
        self,
        replacements,
        remove_tags,
        sample_rate=16000,
        max_duration=float("inf"),
        max_transcript_len=float("inf"),
        min_duration=0.0,
        normalize_transcripts=NormalizeLevel.IDENTITY,
        standardize_wer=True,
        trim_silence=False,
        speed_perturbation=None,
        user_symbols=None,
        error_rate="word",
        standardize_text=False,
        preserve_case=False,
        preserve_punctuation=False,
        append_EOI=False,
    ):
        pass


class SpeedPerturbationParams:
    def __init__(
        self,
        min_rate=0.85,
        max_rate=1.15,
        p=1.0,
    ):
        pass


def warn(length):
    if length == 0:
        print("WARNING: Audio is silent, consider lowering silence threshold")


class SimpleDaliPipeline(nvidia.dali.pipeline.Pipeline, ABC):
    """
    Base DALI pipeline for audio processing and feature extraction.

    This class provides the common audio processing functionality used across
    different pipeline types (training, validation, inference). It handles the
    core audio transformation steps like preprocessing, spectrogram calculation,
    mel-filterbank application, and normalization.

    This is an abstract class - subclasses must implement read_data_in()
    and define_graph() methods.
    """

    @beartype
    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        seed,
        prefetch_queue_depth,
        exec_async,
        exec_pipelined,
        preprocessing_device,
        dali_yaml_config,
        mel_feat_normalizer,
        turn_off_initial_padding,
        final_padding_secs,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
            exec_async=exec_async,
            exec_pipelined=exec_pipelined,
        )

        self.preprocessing_device = preprocessing_device.lower()
        assert self.preprocessing_device in [
            "cpu",
            "gpu",
        ], "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"

        self.sample_rate = dali_yaml_config.sample_rate
        self.dither_coeff = dali_yaml_config.dither_coeff
        self.normalize = mel_feat_normalizer
        self.do_normalize = bool(mel_feat_normalizer)
        self.append_n_EOI = dali_yaml_config.append_n_EOI

        self.decode = ops.decoders.Audio(
            device="cpu",
            dtype=types.FLOAT,
            downmix=True,
            sample_rate=self.sample_rate,
        )

        window_size = dali_yaml_config.window_size
        window_stride = dali_yaml_config.window_stride
        # window_size and window_stride are measured in seconds
        assert window_size > window_stride

        num_zeros = int(self.sample_rate * (window_size - window_stride))
        # The ASR server pads the start of the audio with this many
        # zeros, so the same is done for training and validation
        self.initial_zeros = np.zeros(num_zeros)
        self.turn_off_initial_padding = turn_off_initial_padding

        self.final_zeros = (
            np.zeros(int(self.sample_rate * final_padding_secs))
            if final_padding_secs > 0
            else None
        )

        self.normal_distribution = ops.random.Normal(device=preprocessing_device)

        self.preemph = ops.PreemphasisFilter(
            device=preprocessing_device, preemph_coeff=dali_yaml_config.preemph_coeff
        )

        stride = int(self.sample_rate * window_stride)
        extent = int(self.sample_rate * window_size)

        self.spectrogram = ops.Spectrogram(
            device=preprocessing_device,
            nfft=dali_yaml_config.nfft,
            window_length=extent,
            window_step=stride,
            center_windows=False,
        )

        self.mel_fbank = ops.MelFilterBank(
            device=preprocessing_device,
            sample_rate=self.sample_rate,
            nfilter=dali_yaml_config.nfeatures,
            normalize=True,
        )

        self.log_features = ops.ToDecibels(
            device=preprocessing_device,
            multiplier=np.log(10),
            reference=1.0,
            cutoff_db=math.log(1e-20),
        )

        self.get_shape = ops.Shapes(device=preprocessing_device)

        self.pad = ops.Pad(device=preprocessing_device, fill_value=0)

        if self.append_n_EOI is not None:
            # Pad to a multiple of this with silence
            self.stride_align = self.append_n_EOI * stride

            # Extend with this much silence so the spectrogram transform
            # doesn't pad
            self.silence_window = np.zeros(extent - stride, dtype=np.float32)

            # Use alternating 1, -1 pattern for EOI token
            self.EOI = np.arange(dali_yaml_config.nfeatures, dtype=np.int32)
            self.EOI = 2 * (self.EOI % 2) - 1
            self.EOI = self.EOI.astype(np.float32)
            self.EOI = np.stack([self.EOI] * self.append_n_EOI, axis=1)

    def common_preprocess(self, audio):
        if not self.turn_off_initial_padding:
            audio = nvidia.dali.fn.cat(self.initial_zeros, audio)

        if self.final_zeros is not None:
            audio = nvidia.dali.fn.cat(audio, self.final_zeros)

        if self.append_n_EOI is not None:
            # Pad to multiple of stride, there will be len / stride samples
            audio = nvidia.dali.fn.pad(
                audio,
                fill_value=0.0,
                axes=(0,),
                shape=(1,),  # Min-shape
                align=(self.stride_align,),
            )
            # Extend such that last sample has a complete window in-bounds
            audio = nvidia.dali.fn.cat(audio, self.silence_window)

        if self.dither_coeff != 0.0:
            audio = audio + self.normal_distribution(audio) * self.dither_coeff

        self._inspect_audio(audio)

        audio = self.preemph(audio)
        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)

        if self.do_normalize:
            audio = self.normalize(audio)

        if self.append_n_EOI is not None:
            # Append End-Of-Input "token" to audio
            audio = nvidia.dali.fn.cat(audio, self.EOI, axis=1)

        audio_len = self.get_shape(audio)
        audio = self.pad(audio)
        return audio, audio_len

    def _inspect_audio(self, audio):
        """
        No-op for subclasses to override.
        """
        return audio

    @abstractmethod
    def read_data_in(self):
        """
        Subclass must implement how audio is read.
        """
        raise NotImplementedError("Subclass must define how data is read.")

    @abstractmethod
    def define_graph(self):
        """
        Subclass must implement the DALI pipeline graph.
        """
        raise NotImplementedError("Subclass must define the DALI pipeline graph.")


class InferenceOnlyPipeline(SimpleDaliPipeline):
    """
    DALI pipeline for inference-only tasks.

    This pipeline provides a streamlined version of audio processing focused on
    inference without training-specific components such as augmentations and labels.

    The pipeline expects raw audio data to be fed through the external source
    and produces processed audio features suitable for model inference.

    Used in myrtle-data/src/asr_filter/clean/rnnt.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = ops.ExternalSource(name="file_source", batch=True)

    def read_data_in(self):
        """
        Read audio data from external source.
        """
        audio = self.read()
        audio, _ = self.decode(audio)
        return audio

    def define_graph(self):
        audio = self.read_data_in()
        if self.preprocessing_device == "gpu":
            audio = audio.gpu()
        audio, audio_len = self.common_preprocess(audio)
        return audio, audio_len


class DaliPipeline(SimpleDaliPipeline):
    """
    Full-featured DALI pipeline for ASR model training and validation.

    This pipeline extends SimpleDaliPipeline with comprehensive functionality
    for audio data augmentation and preprocessing during model training. It
    supports various data sources, multiple augmentation techniques, and
    specialized handling for training versus validation pipelines.
    """

    def __init__(
        self,
        *,
        pipeline_type,
        device_id,
        num_threads: int,
        batch_size,
        file_root: str,
        sampler,
        dali_yaml_config: DaliYAMLConfig,
        noise_augmentation_args: NoiseAugmentationArgs,
        seed: int,
        external_reader: WebDatasetReader | HuggingFaceReader | None,
        data_source: DataSource,
        turn_off_initial_padding: bool,
        inspect_audio: bool,
        prob_narrowband: float,
        output_dir: Path,
        preprocessing_device: str,
        no_logging: bool,
        final_padding_secs: float,
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        self.do_background_noise_aug = (
            noise_augmentation_args.prob_background_noise > 0.0
        )
        self.do_babble_noise_aug = noise_augmentation_args.prob_babble_noise > 0.0
        self.do_microphone_aug = noise_augmentation_args.prob_microphone_aug > 0.0
        self.do_room_aug = noise_augmentation_args.prob_room_aug > 0.0
        pipelined_possible = not inspect_audio
        async_possible = pipelined_possible and not self.do_babble_noise_aug
        super().__init__(
            batch_size,
            num_threads,
            device_id,
            seed,
            prefetch_queue_depth=2,  # default
            exec_async=async_possible,
            exec_pipelined=pipelined_possible,
            preprocessing_device=preprocessing_device,
            dali_yaml_config=dali_yaml_config,
            mel_feat_normalizer=mel_feat_normalizer,
            turn_off_initial_padding=turn_off_initial_padding,
            final_padding_secs=final_padding_secs,
        )

        if not no_logging:
            self._dali_init_log(locals())

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            n_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        resample_range = dali_yaml_config.resample_range
        self.prob_narrowband = prob_narrowband

        train_pipeline = pipeline_type == "train"
        self.train = train_pipeline
        self.data_source = data_source

        silence_threshold = dali_yaml_config.silence_threshold
        self.do_remove_silence = True if silence_threshold is not None else False

        self.inspect_audio = inspect_audio
        if self.inspect_audio:
            self.save_audio = ops.PythonFunction(
                save_audio_factory(pipeline_type, shard_id, output_dir),
                num_outputs=1,
                device="cpu",
                batch_processing=False,
            )

        # noise augmentation
        if self.do_background_noise_aug:
            self.background_noise_iterator = NoiseAugmentationIterator(
                batch_size, noise_augmentation_args.prob_background_noise
            )

            self.noise_source = ops.ExternalSource(
                source=self.background_noise_iterator, num_outputs=2
            )

            time_print_once("Making noise dataset for DALI")
            noise_dir = make_noise_dataset_for_dali(
                noise_augmentation_args.use_noise_audio_folder,
                noise_augmentation_args.noise_dataset,
                noise_config=noise_augmentation_args.noise_config,
                sample_rate=self.sample_rate,
                noise_cache_dir=persistent_cache("noise_cache"),
            )
            time_print_once("Done making noise dataset for DALI")

            self.numpy_noise = ops.readers.Numpy(
                name="NumpyNoiseReader",
                device="cpu",
                file_root=noise_dir,
                shard_id=shard_id,
                num_shards=n_shards,
                pad_last_batch=(not train_pipeline),
                shuffle_after_epoch=train_pipeline,
            )

            self.background_noise_func = NumbaFunction(
                run_fn=blend_dali_api,
                in_types=[
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                ],
                ins_ndim=[1, 1, 1, 1],
                outs_ndim=[1, 1, 1, 1],
                out_types=[
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                    DALIDataType.FLOAT,
                ],
                device="cpu",
            )

        if self.do_babble_noise_aug:
            self.babble_noise_iterator = NoiseAugmentationIterator(
                batch_size, noise_augmentation_args.prob_babble_noise
            )
            self.babble_noise_source = ops.ExternalSource(
                source=self.babble_noise_iterator, num_outputs=2
            )
            # Apply babble on a batch level so that other samples from the
            # batch can be used as babble noise. Hence set batch_processing=True
            self.babble_noise_func = NumbaFunction(
                run_fn=babble_batch_dali_api,
                in_types=[DALIDataType.FLOAT, DALIDataType.FLOAT, DALIDataType.FLOAT],
                ins_ndim=[1, 1, 1],
                outs_ndim=[1, 1, 1],
                out_types=[DALIDataType.FLOAT, DALIDataType.FLOAT, DALIDataType.FLOAT],
                device="cpu",
                batch_processing=True,
            )

        if self.do_microphone_aug:
            self.microphone_noise_fn = ops.PythonFunction(
                ImpulseAug(
                    sample_rate=self.sample_rate,
                    prob=noise_augmentation_args.prob_microphone_aug,
                    impulse_dir=noise_augmentation_args.microphone_dataset,
                    impulse_cache_dir=persistent_cache("microphone_cache"),
                ),
                num_outputs=1,
                device="cpu",
                batch_processing=False,
            )

        if self.do_room_aug:
            self.room_noise_fn = ops.PythonFunction(
                ImpulseAug(
                    sample_rate=self.sample_rate,
                    prob=noise_augmentation_args.prob_room_aug,
                    impulse_dir=noise_augmentation_args.room_dataset,
                    impulse_cache_dir=persistent_cache("room_cache"),
                ),
                num_outputs=1,
                device="cpu",
                batch_processing=False,
            )

        self.reader_name = (
            f"{pipeline_type}-reader" if data_source is DataSource.JSON else None
        )

        if data_source is not DataSource.JSON:
            assert sampler is None, "Sampler not required for WebDataset/Hugging Face"
            self.read = ops.ExternalSource(
                source=external_reader,
                num_outputs=4,
                batch=False,
                parallel=False,
                name=self.reader_name,
                cycle="raise",
            )
        else:
            self.read = ops.readers.File(
                name=self.reader_name,
                pad_last_batch=True,  # Doesn't matter if we never hit the end of shard
                device="cpu",
                file_root=file_root,
                file_list=sampler.file_list_path,
                shard_id=shard_id,
                num_shards=n_shards,
                shuffle_after_epoch=False,  # Shuffle is done in the sampler
            )

        if resample_range is not None:
            self.speed_perturbation_coeffs = ops.random.Uniform(
                device="cpu", range=resample_range
            )
        else:
            self.speed_perturbation_coeffs = None

        self.prob_speed_perturbation = dali_yaml_config.prob_speed_perturbation

        # Silence trimming
        self.get_nonsilent_region = ops.NonsilentRegion(
            device="cpu", cutoff_db=silence_threshold
        )

    @staticmethod
    def _dali_init_log(args: dict):
        if not torch.distributed.is_initialized() or (
            torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
        ):  # print once
            max_len = max([len(ii) for ii in args.keys()])
            fmt_string = "\t%" + str(max_len) + "s : %s"
            print("Initializing DALI with parameters:")
            for keyPair in sorted(args.items()):
                print(fmt_string % keyPair)

    def _remove_silence(self, inp):
        """
        Remove silence from audio.
        """
        # If length == 0 then start is undefined (see docs)
        start, length = self.get_nonsilent_region(inp)

        # Do not use batch processing as it would prevent pipelining
        nvidia.dali.fn.python_function(
            length,
            function=warn,
            preserve=True,
            num_outputs=0,
        )

        fallback_beg = 0
        fallback_end = nvidia.dali.fn.shapes(inp)[0]

        beg = (length == 0) * fallback_beg + (length != 0) * (start)
        end = (length == 0) * fallback_end + (length != 0) * (start + length)

        return inp[beg:end]

    def _inspect_audio(self, audio):
        """
        Save audio if inspection is enabled.
        """
        if self.inspect_audio:
            self.save_audio(audio)
        return audio

    def read_data_in(self):
        """
        Load data from source and decode it. This happens on the CPU.
        The audio's sample rate is converted to self.sample_rate on the
        CPU. If the audio is pre-processed to the correct sample rate then
        this is a noop.
        """

        if self.data_source is DataSource.JSON:
            audio, label = self.read()

            # the 'labels' are just indexes so their lengths == 1 and are meaningless
            label_len = self.get_shape(label)
            audio, sr = self.decode(audio)

            # The last ones are just placeholders
            return audio, label, label_len, 42, 314
        else:
            audio, label, raw_transcript, fname = self.read()

            shape = self.get_shape(label)
            label_len = nvidia.dali.fn.cast(shape, dtype=types.INT32)
            label = nvidia.dali.fn.pad(label, fill_value=0)

            return audio, label, label_len, raw_transcript, fname

    def define_graph(self):
        # Audio's sample rate is self.sample_rate
        audio, label, label_lens, raw_transcript, fname = self.read_data_in()

        if self.do_microphone_aug:
            audio = self.microphone_noise_fn(audio)

        if self.do_room_aug:
            audio = self.room_noise_fn(audio)

        if self.do_remove_silence:
            # This is currently CPU->CPU (could be GPU in/out)
            audio = self._remove_silence(audio)

        if self.do_babble_noise_aug:
            babble_target_snr, babble_ratio_start = self.babble_noise_source()
            # This uses Numba and is currently CPU->CPU
            audio, _, _ = self.babble_noise_func(
                audio, babble_target_snr, babble_ratio_start
            )

        if self.do_background_noise_aug:
            target_snr, ratio_start = self.noise_source()
            # Noise is pre-processed into self.sample_rate
            noise = self.numpy_noise()
            # Noise blend is CPU->CPU as using Numba
            audio, _, _, _ = self.background_noise_func(
                audio, noise, target_snr, ratio_start
            )

        if self.preprocessing_device == "gpu":
            audio = audio.gpu()

        if (
            self.train
            and self.speed_perturbation_coeffs is not None
            and self.prob_speed_perturbation > 0
        ):
            candidate_scale = self.speed_perturbation_coeffs()

            resample_on = nvidia.dali.fn.random.coin_flip(
                probability=self.prob_speed_perturbation, device="cpu"
            )

            # Not using an if statement because DALI would trace only one branch
            resample_scale = (1.0 - resample_on) * 1.0 + resample_on * candidate_scale

            audio = nvidia.dali.fn.audio_resample(
                audio,
                scale=resample_scale,
                quality=100,
                device=self.preprocessing_device,
            )

        if self.prob_narrowband > 0.0:
            # Reduce volume to avoid clipping
            audio = audio / 3.0

            # target_sr will be either self.sample_rate or 8000
            diff_sr = self.sample_rate - 8000.0

            target_sr = self.sample_rate - diff_sr * nvidia.dali.fn.random.coin_flip(
                probability=self.prob_narrowband
            )

            # TODO: one of these resamples could be fused with speed_perturbation resample

            # resample to target_sr
            audio = nvidia.dali.fn.audio_resample(
                audio,
                in_rate=self.sample_rate,
                out_rate=target_sr,
                quality=100,
                device=self.preprocessing_device,
            )
            # resample back to original sample rate
            audio = nvidia.dali.fn.audio_resample(
                audio,
                in_rate=target_sr,
                out_rate=self.sample_rate,
                quality=100,
                device=self.preprocessing_device,
            )

            audio = audio * 3.0

        audio, audio_len = self.common_preprocess(audio)

        raw_transcript = nvidia.dali.fn.pad(raw_transcript, fill_value=-1)
        fname = nvidia.dali.fn.pad(fname, fill_value=-1)

        # When modifying DALI pipeline returns, make sure you update `output_map`
        # in DALIGenericIterator invocation
        # modified to return args on the preprocessing device
        return audio, audio_len, label, label_lens, raw_transcript, fname


def save_audio_factory(pipeline_type, shard_id, output_dir: Path):
    audio_dir = output_dir / "augmented_audios"
    audio_dir.mkdir(parents=True, exist_ok=True)
    i = 0

    def save_audio(np_array):
        nonlocal i
        write(
            audio_dir / f"audio_{pipeline_type}_{shard_id}_{i}.wav",
            16000,
            np_array,
        )
        i += 1
        return np_array

    return save_audio
