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
from pathlib import Path

import numpy as np
import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from beartype.typing import Optional
from nvidia.dali.plugin.numba.experimental import NumbaFunction
from nvidia.dali.types import DALIDataType
from scipy.io.wavfile import write

from caiman_asr_train.args.noise_augmentation import NoiseAugmentationArgs
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.noise import (
    BabbleNoiseIterator,
    BackgroundNoiseIterator,
    babble_batch_dali_api,
    blend_dali_api,
)
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.hugging_face.core import HuggingFaceReader
from caiman_asr_train.data.webdataset import WebDatasetReader
from caiman_asr_train.setup.dali import DaliYAMLConfig


class PipelineParams:
    def __init__(
        self,
        sample_rate=16000,
        max_duration=float("inf"),
        max_transcript_len=float("inf"),
        normalize_transcripts=False,
        standardize_wer=True,
        trim_silence=False,
        speed_perturbation=None,
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


class DaliPipeline(nvidia.dali.pipeline.Pipeline):
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
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        self.do_background_noise_aug = (
            noise_augmentation_args.prob_background_noise > 0.0
        )
        self.do_babble_noise_aug = noise_augmentation_args.prob_babble_noise > 0.0
        pipelined_possible = not inspect_audio
        async_possible = pipelined_possible and not self.do_babble_noise_aug
        super().__init__(
            batch_size,
            num_threads,
            device_id,
            exec_pipelined=pipelined_possible,
            exec_async=async_possible,
            seed=seed,
        )

        if not no_logging:
            self._dali_init_log(locals())

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            n_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        self.preprocessing_device = preprocessing_device.lower()
        assert (
            self.preprocessing_device == "cpu" or self.preprocessing_device == "gpu"
        ), "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"

        resample_range = dali_yaml_config.resample_range
        self.prob_narrowband = prob_narrowband

        train_pipeline = pipeline_type == "train"
        self.train = train_pipeline
        self.data_source = data_source
        self.sample_rate = dali_yaml_config.sample_rate
        self.dither_coeff = dali_yaml_config.dither_coeff
        self.normalize = mel_feat_normalizer
        self.do_normalize = bool(mel_feat_normalizer)

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
            self.background_noise_iterator = BackgroundNoiseIterator(
                batch_size,
                shard_id,
                n_shards,
                noise_augmentation_args.noise_dataset,
                noise_augmentation_args.use_noise_audio_folder,
                noise_augmentation_args.prob_background_noise,
                noise_config=noise_augmentation_args.noise_config,
                sample_rate=self.sample_rate,
            )
            self.noise_source = ops.ExternalSource(
                source=self.background_noise_iterator, num_outputs=3
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
            self.babble_noise_iterator = BabbleNoiseIterator(
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

        if data_source is not DataSource.JSON:
            assert sampler is None, "Sampler not required for WebDataset/Hugging Face"
            self.read = ops.ExternalSource(
                source=external_reader,
                num_outputs=2,
                batch=False,
                parallel=False,
                cycle="raise",
            )
        else:
            shuffle = train_pipeline and not sampler.is_sampler_random()
            self.read = ops.readers.File(
                name="Reader",
                pad_last_batch=(not train_pipeline),
                device="cpu",
                file_root=file_root,
                file_list=sampler.get_file_list_path(),
                shard_id=shard_id,
                num_shards=n_shards,
                shuffle_after_epoch=shuffle,
            )

        # use highest resampling quality = '100'
        self.resample = nvidia.dali.ops.AudioResample(quality=100, dtype=types.FLOAT)

        if resample_range is not None:
            self.speed_perturbation_coeffs = ops.random.Uniform(
                device="cpu", range=resample_range
            )
        else:
            self.speed_perturbation_coeffs = None

        self.decode = ops.decoders.Audio(
            device="cpu",
            sample_rate=self.sample_rate if resample_range is None else None,
            dtype=types.FLOAT,
            downmix=True,
        )

        window_size = dali_yaml_config.window_size
        window_stride = dali_yaml_config.window_stride
        assert window_size > window_stride
        # window_size and window_stride are measured in seconds
        how_many_zeros = self.sample_rate * (window_size - window_stride)
        # The ASR server pads the start of the audio with this many
        # zeros, so the same is done for training and validation
        self.initial_zeros = np.zeros(int(how_many_zeros))
        self.turn_off_initial_padding = turn_off_initial_padding

        self.normal_distribution = ops.random.Normal(device=preprocessing_device)

        self.preemph = ops.PreemphasisFilter(
            device=preprocessing_device, preemph_coeff=dali_yaml_config.preemph_coeff
        )

        self.spectrogram = ops.Spectrogram(
            device=preprocessing_device,
            nfft=dali_yaml_config.nfft,
            window_length=window_size * self.sample_rate,
            window_step=window_stride * self.sample_rate,
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

        # Silence trimming
        self.get_nonsilent_region = ops.NonsilentRegion(
            device="cpu", cutoff_db=silence_threshold
        )
        self.trim_silence = ops.Slice(
            device="cpu", normalized_anchor=False, normalized_shape=False, axes=[0]
        )
        self.to_float = ops.Cast(device="cpu", dtype=types.FLOAT)
        self.to_int = ops.Cast(device="cpu", dtype=types.INT32)

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
        begin, length = self.get_nonsilent_region(inp)
        out = self.trim_silence(inp, self.to_float(begin), self.to_float(length))
        return out

    def read_data_in(self, resample_scale):
        audio, label = self.read()

        # 1) deal with labels and their lengths
        if self.data_source is not DataSource.JSON:
            label_len = self.to_int(self.get_shape(label))
            label = self.pad(label)
        else:
            # the 'labels' are just indexes so their lengths == 1 and are meaningless
            label_len = self.get_shape(label)

        # 2) decode audio (already done in tar-file/Hugging Face cases)
        if self.data_source is DataSource.JSON:
            audio, sr = self.decode(audio)

        if resample_scale is not None:
            audio = self.resample(audio, scale=resample_scale)

        return audio, label, label_len

    def define_graph(self):
        if self.train and self.speed_perturbation_coeffs is not None:
            resample_scale = self.speed_perturbation_coeffs()
        else:
            resample_scale = None

        audio, label, label_lens = self.read_data_in(resample_scale)

        if self.do_background_noise_aug:
            noise, target_snr, ratio_start = self.noise_source()
            if resample_scale is None:
                noise, nr = self.decode(noise)
            else:
                resample_coeffs = resample_scale * self.sample_rate
                noise, nr = self.decode(noise, sample_rate=resample_coeffs)

        if self.do_remove_silence:
            audio = self._remove_silence(audio)

        if self.do_babble_noise_aug:
            babble_target_snr, babble_ratio_start = self.babble_noise_source()
            audio, _, _ = self.babble_noise_func(
                audio, babble_target_snr, babble_ratio_start
            )

        if self.do_background_noise_aug:
            # blend noise and audio here
            audio, _, _, _ = self.background_noise_func(
                audio, noise, target_snr, ratio_start
            )

        if self.prob_narrowband > 0.0:
            # Reduce volume to avoid clipping
            audio = audio / 3.0

            # target_sr will be either self.sample_rate or 8000
            diff_sr = self.sample_rate - 8000.0
            target_sr = self.sample_rate - diff_sr * nvidia.dali.fn.random.coin_flip(
                probability=self.prob_narrowband
            )

            # resample to target_sr
            audio = nvidia.dali.fn.audio_resample(
                audio, in_rate=self.sample_rate, out_rate=target_sr, quality=100
            )
            # resample back to original sample rate
            audio = nvidia.dali.fn.audio_resample(
                audio, in_rate=target_sr, out_rate=self.sample_rate, quality=100
            )

            audio = audio * 3.0

        if self.preprocessing_device == "gpu":
            audio = audio.gpu()

        if not self.turn_off_initial_padding:
            audio = nvidia.dali.fn.cat(self.initial_zeros, audio)

        if self.dither_coeff != 0.0:
            audio = audio + self.normal_distribution(audio) * self.dither_coeff

        if self.inspect_audio:
            audio = self.save_audio(audio)

        audio = self.preemph(audio)

        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)

        audio_len = self.get_shape(audio)

        if self.do_normalize:
            audio = self.normalize(audio)

        audio = self.pad(audio)

        # When modifying DALI pipeline returns, make sure you update `output_map`
        # in DALIGenericIterator invocation
        # modified to return args on the preprocessing device
        return audio, audio_len, label, label_lens


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
