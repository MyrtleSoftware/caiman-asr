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

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.train import train_arg_parser, verify_train_args
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.noise import NoiseSchedule
from caiman_asr_train.evaluate import calculate_wer, evaluate
from caiman_asr_train.log.logging_layers import get_logging_entries
from caiman_asr_train.log.profiling import (
    finish_profiling,
    save_timings,
    set_up_profiling,
)
from caiman_asr_train.log.tb_dllogger import flush_log, log
from caiman_asr_train.setup.base import TRAIN, VAL
from caiman_asr_train.setup.train import TrainSetup
from caiman_asr_train.train_utils.core import calculate_epoch, log_end_of_epoch
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.rsp import (
    generate_batch_history,
    rsp_config_checks,
    rsp_end_step,
)
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, "module", model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


def main(args, train_objects):
    args = verify_train_args(args)
    check_shared_args(args)
    world_size = train_objects.world_size

    assert (
        args.prediction_frequency is None
        or args.prediction_frequency % args.log_frequency == 0
    )
    cfg = train_objects.cfg

    rsp_config_checks(args, cfg)

    assert args.grad_accumulation_batches >= 1

    out_dir = Path(args.output_dir)
    # fail if output dir already contains checkpoints
    if not args.resume and out_dir.exists() and any(out_dir.glob("*checkpoint*.pt")):
        error_msg = (
            f"{out_dir=} already contains checkpoints which would be overwritten by this "
            "command. Running training using the same output_dir as a previous command "
            "is only permitted when args.resume=True."
        )
        if args.fine_tune:
            error_msg += (
                " In the args.fine_tune=True case it is recommended to pass args.ckpt "
                "of the form /checkpoints/<ckpt_path> instead of /results/<ckpt_path> in "
                "order to avoid this error."
            )
        raise ValueError(error_msg)

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

    assert (
        initial_step < args.training_steps
    ), f"{initial_step=} and {args.training_steps=}. No training to do. Exiting."

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

    rsp_counter = generate_batch_history(args.rsp_seq_len_freq)

    # training loop
    while step <= args.training_steps:
        mel_feat_norm.step(step)
        if noise_schedule is not None:
            noise_schedule.adjust_snrs(step)

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
                all_feat_lens = []
                losses = []
                if noise_schedule is not None:
                    noise_schedule.adjust_snrs(step)
                mel_feat_norm.step(step)

            audio, audio_lens, txt, txt_lens = batch

            # audio is (batch, meldim, max_len)
            # audio_lens is (batch,)

            # if these tensors were computed on cpu then move them to gpu
            if args.dali_device == "cpu":
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
            )

            forward_backward_total += time.time() - before_forward_backward

            if not loss_nan:
                losses.append(loss_item)
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1

            state, rsp_counter = rsp_end_step(state, loss_nan, step, args, rsp_counter)

            # the > 0 condition prevents 1st batch NaNs entering this code
            if (
                accumulated_batches > 0
                and accumulated_batches % args.grad_accumulation_batches == 0
            ):
                total_norm, tb_per_layer_logs, log2_scaler = get_logging_entries(
                    model, scaler
                )

                # Add noise to the gradients of the encoder tensors
                grad_noise_scheduler = train_objects.training_only.grad_noise_scheduler
                if grad_noise_scheduler is not None:
                    model = grad_noise_scheduler.add_grad_noise(
                        model=model,
                        step=step,
                        world_size=world_size,
                    )
                optimizer_wrapper.step(total_norm)

                apply_ema(model, ema_model, args.ema)

                if step % args.log_frequency == 0:
                    if (
                        args.prediction_frequency is None
                        or step % args.prediction_frequency == 0
                    ):
                        preds, _ = decoder.decode(
                            model, feats, feat_lens, args.max_inputs_per_batch
                        )
                        wer, pred_utt, ref = calculate_wer(
                            preds,
                            txt,
                            txt_lens,
                            tokenizer.detokenize,
                            standardize_wer=train_standardize_wer,
                        )
                        print_once(f"  Decoded:   {pred_utt[:90]}")
                        print_once(f"  Reference: {ref[:90]}")
                        wer = {"wer": 100 * wer}
                    else:
                        wer = {}

                    # log mel statistics for logging
                    audio_no_pad = [
                        audio[i, :, : audio_lens[i]] for i in range(audio.shape[0])
                    ]
                    no_pad_tensor = torch.cat(audio_no_pad, dim=1)
                    no_pad_mean = no_pad_tensor.mean()
                    no_pad_std = no_pad_tensor.std()

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
                            "throughput": step_utts / step_time,
                            "took": step_time,
                            "grad-norm": total_norm,
                            "seq-len-min": min(all_feat_lens).item(),
                            "seq-len-max": max(all_feat_lens).item(),
                            "seq-len-mean": (
                                sum(all_feat_lens) / len(all_feat_lens)
                            ).item(),
                            "lrate": lr_,
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
                if step % args.val_frequency == 0 or step == args.training_steps:
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
                        calculate_loss=True,
                        using_cpu=False,
                    )["wer"]

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
                        )

                # saving checkpoint
                save_this_step = (
                    bool(args.save_frequency) and step % args.save_frequency == 0
                )
                save_this_step = save_this_step or (
                    step == args.training_steps and not args.dont_save_at_the_end
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
                    )

                # end of step
                step += 1
                if step > args.training_steps:
                    break
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
    train_objects = TrainSetup().run(args)
    main(args, train_objects)
    finish_profiling(args.profiler, args.output_dir, profilers, args.timestamp)
