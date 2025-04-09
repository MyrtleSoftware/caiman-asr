#! /usr/bin/env python3
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import json
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from beartype import beartype

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.train import train_arg_parser, verify_train_args
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.noise import NoiseSchedule
from caiman_asr_train.evaluate.core import calculate_wer, evaluate
from caiman_asr_train.evaluate.error_rates import error_rate_abbrev
from caiman_asr_train.log.logging_layers import get_logging_entries
from caiman_asr_train.log.profiling import (
    finish_profiling,
    save_timings,
    set_up_profiling,
)
from caiman_asr_train.log.tb_dllogger import flush_log, log
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.loss import LossModifiers
from caiman_asr_train.setup.core import TRAIN, VAL
from caiman_asr_train.setup.train import TrainSetup
from caiman_asr_train.train_utils.core import calculate_epoch, log_end_of_epoch
from caiman_asr_train.train_utils.distributed import (
    print_once,
    time_print_once,
    unwrap_ddp,
)
from caiman_asr_train.train_utils.rsp import (
    generate_batch_history,
    rsp_config_checks,
    rsp_end_step,
)
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun
from caiman_asr_train.utils.frame_width import input_feat_frame_width
from caiman_asr_train.utils.responses import split_batched_finals


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = unwrap_ddp(model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


@beartype
def open_logs(args) -> tuple:
    id = dist.get_rank() if dist.is_initialized() else 0

    files = [
        f"log_utt_lens_{id}_{args.timestamp}.txt",
        f"log_utt_file_{id}_{args.timestamp}.txt",
    ]

    files = map(Path, files)

    files = map(lambda p: args.output_dir / p, files)

    return tuple(map(lambda p: p.open("w"), files))


def main(args):
    args = verify_train_args(args)
    check_shared_args(args)

    train_objects = TrainSetup().run(args)
    time_print_once("Done with training setup")

    world_size = train_objects.world_size

    assert (
        args.prediction_frequency is None
        or args.prediction_frequency % args.log_frequency == 0
    )
    cfg = train_objects.cfg

    rsp_config_checks(args, cfg)

    assert args.grad_accumulation_batches >= 1

    out_dir = Path(args.output_dir)

    if world_size == 1 or dist.get_rank() == 0:
        # save configuration to file
        path_to_saved_args = out_dir / f"training_args_{args.timestamp}.json"
        nicely_formatted_args = json.dumps(vars(args), indent=2)
        path_to_saved_args.write_text(nicely_formatted_args)

        saved_config_path = (
            out_dir / f"{Path(args.model_config).stem}_{args.timestamp}.yaml"
        )
        shutil.copy(args.model_config, saved_config_path)

    if train_objects.multi_gpu:
        torch.distributed.barrier()

    train_loader = train_objects.data_objects[TRAIN].loader
    val_loader = train_objects.data_objects[VAL].loader

    meta = train_objects.training_only.meta
    epoch = meta["start_epoch"]
    best_wer = meta["best_wer"]
    step = initial_step = meta["step"]

    steps_per_epoch = None
    if args.resume:
        step += 1
    if not args.read_from_tar:
        # steps per epoch is unknown for tarred data
        steps_per_epoch = len(train_loader) // args.grad_accumulation_batches
        # update epoch if next step is a new epoch
        epoch = calculate_epoch(step, steps_per_epoch)

    train_standardize_wer = train_objects.data_objects[TRAIN].dataset_kw[
        "standardize_wer"
    ]
    val_standardize_wer = train_objects.data_objects[VAL].dataset_kw["standardize_wer"]

    noise_schedule = None
    if (
        train_loader.pipeline.do_background_noise_aug
        or train_loader.pipeline.do_babble_noise_aug
    ):
        noise_schedule = NoiseSchedule(
            args.noise_delay_steps,
            args.noise_ramp_steps,
            args.noise_initial_low,
            args.noise_initial_high,
            train_loader,
        )

    mel_feat_norm: MelFeatNormalizer = train_loader.pipeline.normalize

    optimizer_wrapper = train_objects.training_only.optimizer_wrapper
    train_step_fn = train_objects.training_only.train_step_fn
    optimizer = train_objects.training_only.optimizer_wrapper.optimizer
    model = train_objects.model
    train_feat_proc = train_objects.feat_procs[TRAIN]
    val_feat_proc = train_objects.feat_procs[VAL]
    loss_fn = train_objects.loss_fn
    scaler = train_objects.training_only.optimizer_wrapper.scaler
    ema_model = train_objects.ema_model
    tokenizer = train_objects.tokenizers[TRAIN]
    tokenizer_kw = train_objects.tokenizers_kw[TRAIN]
    decoder = train_objects.decoder
    checkpointer = train_objects.training_only.checkpointer
    model.train()
    state = None
    frame_width = input_feat_frame_width(config.load(args.model_config))

    rsp_counter = generate_batch_history(args.rsp_seq_len_freq)

    dp_scheduler = train_objects.training_only.dp_scheduler
    star_scheduler = train_objects.training_only.star_scheduler
    wer_is_bad = False

    if args.log_verbose_utterance_statistics:
        log_lens, log_utts = open_logs(args)

    # training loop
    while step <= args.training_steps:
        mel_feat_norm.step(step)
        if noise_schedule is not None:
            bg_snrs, bb_snrs = noise_schedule.adjust_snrs(step)

        if train_loader.pipeline.do_background_noise_aug:
            print_once(
                f"At start of epoch {epoch} SNRs are "
                f"{train_loader.pipeline.background_noise_iterator.low}-"
                f"{train_loader.pipeline.background_noise_iterator.high} dB"
            )
        if train_loader.pipeline.do_babble_noise_aug:
            print_once(
                f"At start of epoch {epoch} babble SNRs are "
                f"{train_loader.pipeline.babble_noise_iterator.low}-"
                f"{train_loader.pipeline.babble_noise_iterator.high} dB"
            )

        epoch_utts = 0
        accumulated_batches = 0
        step_start_time = time.time()

        dataloading_total = 0.0  # Time spent getting audio from DALI
        feat_proc_total = 0.0  # Time spent in spec augment / frame stacking
        forward_backward_total = 0.0  # Time spent in forward / backward passes

        epoch_start_time = time.time()

        before_dataloading = time.time()

        for batch in train_loader:
            dataloading_total += time.time() - before_dataloading

            if accumulated_batches == 0:
                train_objects.training_only.adjust_lr(step)
                optimizer_wrapper.zero_grad()
                step_utts = 0
                step_frames = 0
                all_feat_lens = []
                losses = []
                if noise_schedule is not None:
                    bg_snrs, bb_snrs = noise_schedule.adjust_snrs(step)
                mel_feat_norm.step(step)

                dp_scheduler.step(step, hints={"wer": best_wer})
                star_scheduler.step(step, hints={"wer": best_wer})

            audio, audio_lens, txt, txt_lens, _, utt_files = batch
            # audio is (batch, meldim, max_len)
            # audio_lens is (batch,)

            if args.log_verbose_utterance_statistics:
                log_lens.write(f"{audio_lens.cpu().tolist()}\n")
                log_utts.write(f"{utt_files}\n")

            # if these tensors were computed on cpu then move them to gpu
            if args.dali_train_device == "cpu":
                audio = audio.cuda()
                audio_lens = audio_lens.cuda()
                txt = txt.cuda()
                txt_lens = txt_lens.cuda()

            before_feat_proc = time.time()

            # now do spec augment / frame stacking
            # feats is (seq_len, batch, input_dim)
            feats, feat_lens = train_feat_proc([audio, audio_lens])
            all_feat_lens += feat_lens

            feat_proc_total += time.time() - before_feat_proc

            before_forward_backward = time.time()

            loss_item, loss_nan, state = train_step_fn(
                model=model,
                loss_fn=loss_fn,
                args=args,
                feats=feats,
                feat_lens=feat_lens,
                txt=txt,
                txt_lens=txt_lens,
                scaler=scaler,
                rnnt_state=state,
                loss_mods=LossModifiers(
                    delay_penalty=dp_scheduler.value(),
                    star_penalty=star_scheduler.value(),
                    eos_penalty=args.eos_penalty,
                ),
            )

            forward_backward_total += time.time() - before_forward_backward

            if not loss_nan:
                losses.append(loss_item)
                step_utts += batch[0].size(0) * world_size
                step_frames += feat_lens.sum()
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1
            else:
                # NaN's pollute accumulated gradients,
                # this will trigger a reset on the next iteration.
                accumulated_batches = 0
                print_once("WARNING: loss is NaN; dropping global batch")

            state, rsp_counter, rsp_on = rsp_end_step(
                state, loss_nan, step, args, rsp_counter
            )

            if accumulated_batches == args.grad_accumulation_batches:
                total_norm, tb_per_layer_logs, log2_scaler = get_logging_entries(
                    model, scaler
                )

                # Add noise to the gradients of the encoder tensors
                grad_noise_scheduler = train_objects.training_only.grad_noise_scheduler

                if grad_noise_scheduler is not None:
                    model, noise_level = grad_noise_scheduler.add_grad_noise(
                        model=model,
                        step=step,
                        world_size=world_size,
                    )
                else:
                    noise_level = 0

                optimizer_wrapper.step(total_norm)

                apply_ema(model, ema_model, args.ema)

                abbrev = error_rate_abbrev(train_objects.error_rate)

                if step % args.log_frequency == 0:
                    if (
                        args.prediction_frequency is None
                        or step % args.prediction_frequency == 0
                    ):
                        preds, _, _ = split_batched_finals(
                            decoder.decode(feats, feat_lens)
                        )

                        wer, pred_utt, ref = calculate_wer(
                            preds,
                            txt,
                            txt_lens,
                            tokenizer.detokenize,
                            standardize_wer=train_standardize_wer,
                            error_rate=train_objects.error_rate,
                        )
                        print_once(f"  Decoded:   {pred_utt[:90]}")
                        print_once(f"  Reference: {ref[:90]}")
                        wer = {abbrev: 100 * wer}
                    else:
                        wer = {}

                    if noise_schedule is not None:
                        noise = {
                            "noise/background_snr_lo": bg_snrs[0],
                            "noise/background_snr_hi": bg_snrs[1],
                            "noise/babble_snr_lo": bb_snrs[0],
                            "noise/babble_snr_hi": bb_snrs[1],
                        }
                    else:
                        noise = {}

                    # log mel statistics for logging
                    audio_no_pad = [
                        audio[i, :, : audio_lens[i]] for i in range(audio.shape[0])
                    ]
                    no_pad_tensor = torch.cat(audio_no_pad, dim=1)
                    no_pad_mean = no_pad_tensor.mean()
                    no_pad_std = no_pad_tensor.std()

                    if dist.is_initialized():
                        dist.all_reduce(step_frames)

                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    # dllogger.log expects a tuple of:
                    # (epoch, <steps in current epoch>, steps_per_epoch, step)
                    # if steps_per_epoch is None (meaning that data is loaded from tar
                    # files and training is on the first epoch) then, logging
                    # is done with  a tuple of the form:
                    # (epoch, <steps in current epoch>, 'unk')
                    if steps_per_epoch is not None:
                        step_in_epoch = step % steps_per_epoch or steps_per_epoch
                        log_step_tuple = (epoch, step_in_epoch, steps_per_epoch, step)
                    else:
                        log_step_tuple = (epoch, step - initial_step + 1, "unk", step)
                    lr_ = optimizer_wrapper.learning_rate
                    log(
                        log_step_tuple,
                        step,
                        "train",
                        {
                            "loss": sum(losses),
                            **wer,  # optional entry
                            "noise/gradient": noise_level,
                            "throughput-audio-samples-per-sec": step_utts / step_time,
                            "throughput-audio-secs-per-sec": step_frames.item()
                            * frame_width
                            / step_time,
                            "took": step_time,
                            "grad-norm": total_norm,
                            "seq-len-min": min(all_feat_lens).item(),
                            "seq-len-max": max(all_feat_lens).item(),
                            "seq-len-mean": (
                                sum(all_feat_lens) / len(all_feat_lens)
                            ).item(),
                            "lrate": lr_,
                            "delay_penalty": dp_scheduler.value(),
                            "star_penalty": star_scheduler.value(),
                            "rsp_on": 1 if rsp_on else -1,
                            **noise,
                            "log2_scaler": log2_scaler,
                            "logmel_mean": no_pad_mean.item(),
                            "logmel_std": no_pad_std.item(),
                            "logmel_norm_weight": mel_feat_norm.dataset_to_utt_ratio,
                            **dict(tb_per_layer_logs),
                        },
                    )
                else:
                    step_start_time = time.time()

                # evaluating on validation set
                if (
                    step == 1
                    or step % args.val_frequency == 0
                    or step == args.training_steps
                ):
                    wer = evaluate(
                        epoch,
                        step,
                        val_loader,
                        val_feat_proc,
                        tokenizer.detokenize,
                        ema_model,
                        loss_fn,
                        decoder,
                        cfg=cfg,
                        args=args,
                        standardize_wer=val_standardize_wer,
                        calculate_loss=not args.skip_val_loss,
                        using_cpu=False,
                        error_rate=train_objects.error_rate,
                    )[abbrev]

                    if args.die_if_wer_bad and wer > 0.99 and step >= 10000:
                        wer_is_bad = True

                    if wer < best_wer:
                        best_wer = wer
                        checkpointer.save(
                            model,
                            ema_model,
                            optimizer,
                            epoch,
                            step,
                            best_wer,
                            tokenizer_kw,
                            mel_feat_norm.dataset_to_utt_ratio,
                            is_best=True,
                            config_path=args.model_config,
                        )

                # saving checkpoint
                save_this_step = (
                    bool(args.save_frequency) and step % args.save_frequency == 0
                )
                save_this_step = save_this_step or (
                    step == args.training_steps and not args.dont_save_at_the_end
                )
                save_this_step = save_this_step or (
                    wer_is_bad and not args.dont_save_at_the_end
                )
                if save_this_step:
                    checkpointer.save(
                        model,
                        ema_model,
                        optimizer,
                        epoch,
                        step,
                        best_wer,
                        tokenizer_kw,
                        mel_feat_norm.dataset_to_utt_ratio,
                        config_path=args.model_config,
                    )

                # end of step
                step += 1
                if step > args.training_steps:
                    checkpointer.save(
                        model,
                        ema_model,
                        optimizer,
                        epoch,
                        step - 1,
                        best_wer,
                        tokenizer_kw,
                        mel_feat_norm.dataset_to_utt_ratio,
                        is_last=True,
                        config_path=args.model_config,
                    )
                    break
                if wer_is_bad:
                    raise ValueError("WER is mysteriously bad")
                accumulated_batches = 0

            before_dataloading = time.time()

        log_end_of_epoch(epoch_start_time, epoch, epoch_utts)

        if steps_per_epoch is not None:
            # update epoch unless in 1st epoch of tar files
            epoch += 1
        else:
            # after running a full epoch, the number of steps is known
            steps_per_epoch = step - initial_step
            # update number of epoch as it hasn't been known for tar files
            epoch = calculate_epoch(step, steps_per_epoch)

        save_timings(
            dataloading_total,
            feat_proc_total,
            forward_backward_total,
            args.output_dir,
            step,
            args.timestamp,
        )

    if args.log_verbose_utterance_statistics:
        log_lens.close()
        log_utts.close()

    flush_log()


if __name__ == "__main__":
    parser = train_arg_parser()
    args = parser.parse_args()
    maybe_restart_with_torchrun(
        args.num_gpus,
        args.called_by_torchrun,
        "/workspace/training/caiman_asr_train/train.py",
    )
    profilers = set_up_profiling(args.profiler, args.output_dir, args.timestamp)
    main(args)
    finish_profiling(args.profiler, args.output_dir, profilers, args.timestamp)
