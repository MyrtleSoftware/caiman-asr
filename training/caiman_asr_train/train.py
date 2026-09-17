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
from beartype.typing import Optional

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.train import train_arg_parser, verify_train_args
from caiman_asr_train.evaluate.core import calculate_per, calculate_wer, evaluate
from caiman_asr_train.evaluate.error_rates import error_rate_abbrev
from caiman_asr_train.keywords.trie import unwrap
from caiman_asr_train.lhotse.optimize_buckets import (
    gather_utt_stats,
    partition_into_buckets,
)
from caiman_asr_train.log.logging_layers import get_logging_entries
from caiman_asr_train.log.profiling import (
    finish_profiling,
    save_timings,
    set_up_profiling,
)
from caiman_asr_train.log.tb_dllogger import flush_log, log
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.loss import LossModifiers
from caiman_asr_train.rnnt.model import StateRNNT
from caiman_asr_train.rnnt.state import scale_state
from caiman_asr_train.setup.core import TRAIN, VAL
from caiman_asr_train.setup.train import TrainSetup
from caiman_asr_train.train_utils.batch_splitting import train_step
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
    ]

    files = map(Path, files)

    files = map(lambda p: args.output_dir / p, files)

    return tuple(map(lambda p: p.open("w"), files))


def main(args):
    args = verify_train_args(args)
    check_shared_args(args)

    train_objects = TrainSetup().run(args)
    time_print_once("Done with training setup")
    assert train_objects.training_only is not None

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

    optimizer_wrapper = train_objects.training_only.optimizer_wrapper
    optimizer = train_objects.training_only.optimizer_wrapper.optimizer
    model = train_objects.model
    val_joint_loss = train_objects.val_joint_loss
    scaler = train_objects.training_only.optimizer_wrapper.scaler
    ema_model = train_objects.ema_model
    tokenizer = train_objects.tokenizers[TRAIN]
    tokenizer_kw = train_objects.tokenizers_kw[TRAIN]
    decoder = train_objects.decoder
    checkpointer = train_objects.training_only.checkpointer
    model.train()
    state: Optional[StateRNNT] = None
    frame_width = input_feat_frame_width(config.load(args.model_config))

    rsp_counter = generate_batch_history(args.rsp_seq_len_freq)

    dp_scheduler = train_objects.training_only.dp_scheduler
    star_scheduler = train_objects.training_only.star_scheduler
    wer_is_bad = False

    if args.log_verbose_utterance_statistics:
        log_lens = open_logs(args)

    if args.optimize_buckets:
        assert world_size == 1, "Bucket optimization only supported for single GPU"

        print_once("Computing optimal buckets...")

        stats = gather_utt_stats(
            config=train_objects.data_objects[TRAIN].config, tokenizer=tokenizer
        )

        n_dur = args.num_dur_buckets
        n_tok = args.num_tok_buckets

        bins = partition_into_buckets(stats, n_dur, n_tok)

        print_once("Writing optimal buckets to disk")

        with (out_dir / f"bins_{n_dur}.json").open("w") as f:
            json.dump(bins.bins_1D, f, indent=2)

        with (out_dir / f"bins_{n_dur}_{n_tok}.json").open("w") as f:
            json.dump(bins.bins_2D, f, indent=2)

        exit(0)

    total_duration = meta["duration"]
    batch_duration = 0.0
    batch_utts = 0

    # training loop
    epoch_utts = 0
    accumulated_batches = 0
    nonfinite_loss_batches = 0  # global batches dropped for a non-finite loss
    skipped_optimizer_steps = 0  # steps the scaler/wrapper refused to apply
    step_start_time = time.time()

    dataloading_total = 0.0  # Time spent getting audio from dataloader
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
            batch_duration = 0.0
            batch_utts = 0

            dp_scheduler.step(step, hints={"wer": best_wer})
            star_scheduler.step(step, hints={"wer": best_wer})

        pipe_meta, cuts, feats, tokens, prefixes = batch

        tokens = unwrap(tokens)

        # If the train has requested CPU-lhotse the data will not be on GPU yet
        if args.loader_train_device == "cpu":
            feats = feats.to("cuda", non_blocking=True)
            tokens = tokens.to("cuda", non_blocking=True)

            if prefixes is not None:
                prefixes = prefixes.to("cuda", non_blocking=True)

        if args.log_verbose_utterance_statistics:
            log_lens.write(f"{feats.lens.cpu().tolist()}\n")

        all_feat_lens += feats.lens

        # Approximate across all GPUs
        total_duration += pipe_meta.batch_dur / 3600 * world_size

        batch_duration += pipe_meta.batch_dur / 3600
        batch_utts += len(cuts)

        before_forward_backward = time.time()

        if state is not None:
            # Make RNN-T state batch size match current batch size
            state = scale_state(len(cuts), state)

        loss_mods = LossModifiers(
            dp_lam=dp_scheduler.value(),
            star_penalty=star_scheduler.value(),
            eos_lam=args.eos_penalty,
        )

        # DDP all-reduces gradients only on the final accumulation micro-batch (no_sync
        # until then), so each parameter is synchronised once per optimizer step.
        is_last_microbatch = accumulated_batches + 1 == args.grad_accumulation_batches

        loss_item, loss_nan, state = train_step(
            model=model,
            args=args,
            feats=feats,
            tokens=tokens,
            prefix=prefixes,
            scaler=scaler,
            rnnt_state=state,
            loss_mods=loss_mods,
            is_last_microbatch=is_last_microbatch,
        )

        forward_backward_total += time.time() - before_forward_backward

        if not loss_nan:
            losses.append(loss_item)
            step_utts += len(cuts) * world_size
            step_frames += feats.lens.sum()
            epoch_utts += len(cuts) * world_size
            accumulated_batches += 1
        else:
            # Non-finite losses pollute the accumulated gradients (in the
            # batch-split path backward has already run when the loss check
            # fires), so discard them here and restart the accumulation
            # window, rather than relying on the zero_grad at the top of the
            # next loop iteration to clean up before anything touches .grad.
            accumulated_batches = 0
            optimizer_wrapper.zero_grad()
            nonfinite_loss_batches += 1
            print_once(
                f"WARNING: loss is non-finite; dropping global batch "
                f"(total dropped: {nonfinite_loss_batches})"
            )

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
                    scale=1.0 if scaler is None else scaler.get_scale(),
                )
            else:
                noise_level = 0

            stepped = optimizer_wrapper.step(total_norm)

            if stepped:
                # Blend only after a real weight update.
                apply_ema(model, ema_model, args.ema)
            else:
                skipped_optimizer_steps += 1

            abbrev = error_rate_abbrev(train_objects.error_rate)

            if step % args.log_frequency == 0:
                if (
                    args.prediction_frequency is None
                    or step % args.prediction_frequency == 0
                ):
                    preds, _, _, _ = split_batched_finals(decoder.decode(feats))

                    wer, pred_utt, ref = calculate_wer(
                        preds,
                        tokens.data.transpose(0, 1),
                        tokens.lens,
                        tokenizer.detokenize,
                        args.val_standardizer_lang,
                        error_rate=train_objects.error_rate,
                    )
                    wer = {abbrev: 100 * wer}
                    if args.calculate_per:
                        per = calculate_per(
                            preds,
                            tokens.data.transpose(0, 1),
                            tokens.lens,
                            tokenizer.detokenize,
                            punctuation_marks=args.punctuation_marks,
                            punctuation_normalization=args.punctuation_normalization,
                        )
                        per = {"per": 100 * per}
                    else:
                        per = {}

                    print_once(f"  Decoded:   {pred_utt[:90]}")
                    print_once(f"  Reference: {ref[:90]}")
                else:
                    wer = {}
                    per = {}

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
                        "total_duration": total_duration,
                        "per_gpu_batch_duration": batch_duration,
                        "per_gpu_batch_size": batch_utts,
                        **wer,  # optional entry
                        **per,
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
                        "dp_lam": dp_scheduler.value(),
                        "star_penalty": star_scheduler.value(),
                        "rsp_on": 1 if rsp_on else -1,
                        "log2_scaler": log2_scaler,
                        "nonfinite-loss-batches": nonfinite_loss_batches,
                        "skipped-optimizer-steps": skipped_optimizer_steps,
                        "logmel_norm_weight": pipe_meta.mel_alpha,
                        **dict(tb_per_layer_logs),
                    },
                )
            else:
                step_start_time = time.time()

            @beartype
            def save(*, is_best: bool = False, is_last: bool = False) -> None:
                checkpointer.save(
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=step,
                    duration=total_duration,
                    best_wer=best_wer,
                    tokenizer_kw=tokenizer_kw,
                    logmel_norm_weight=(
                        0.0 if pipe_meta.mel_alpha is None else pipe_meta.mel_alpha
                    ),
                    is_best=is_best,
                    is_last=is_last,
                    config_path=args.model_config,
                )

            # evaluating on validation set
            if (
                (step == 1 and not args.skip_first_val_eval)
                or step % args.val_frequency == 0
                or step == args.training_steps
                or (step == args.training_steps and not args.skip_final_val_eval)
            ):
                evaluate_results = evaluate(
                    epoch=epoch,
                    step=step,
                    val_loader=val_loader,
                    detokenize=tokenizer.detokenize,
                    ema_model=ema_model,
                    joint_loss=val_joint_loss,
                    decoder=decoder,
                    cfg=cfg,
                    args=args,
                    calculate_loss=not args.skip_val_loss,
                    nth_batch_only=args.val_nth_batch_only,
                    using_cpu=False,
                    error_rate=train_objects.error_rate,
                    loss_mods=loss_mods,
                )
                wer = evaluate_results[abbrev]

                if args.die_if_wer_bad and wer > 0.99 and step >= 10000:
                    wer_is_bad = True

                if wer < best_wer:
                    best_wer = wer
                    save(is_best=True)

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
                save()

            # end of step
            step += 1
            if step > args.training_steps:
                save(is_last=True)
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
        0.0,
        forward_backward_total,
        args.output_dir,
        step,
        args.timestamp,
    )

    if args.log_verbose_utterance_statistics:
        log_lens.close()

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
