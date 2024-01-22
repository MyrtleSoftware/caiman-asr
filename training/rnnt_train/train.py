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

# modified by myrtle

import argparse
import json
import os
import shutil
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rnnt_train.common.evaluate import evaluate
from rnnt_train.common.helpers import greedy_wer, print_once
from rnnt_train.common.logging_layers import get_logging_entries
from rnnt_train.common.rsp import (
    generate_batch_history,
    rsp_config_checks,
    rsp_end_step,
)
from rnnt_train.common.tb_dllogger import flush_log, log
from rnnt_train.setup.base import TRAIN, VAL
from rnnt_train.setup.train import TrainSetup


def train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RNN-T Training Reference")

    training = parser.add_argument_group("training setup")
    training.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs for the entire training",
    )
    training.add_argument(
        "--warmup_steps",
        default=1632,
        type=int,
        help="Initial steps of increasing learning rate",
    )
    training.add_argument(
        "--hold_steps",
        default=18000,
        type=int,
        help="Constant max learning rate steps after warmup",
    )
    training.add_argument(
        "--half_life_steps",
        default=2805,
        type=int,
        help="half life (in steps) for exponential learning rate decay",
    )
    training.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=True,
        help="Enable cudnn benchmark",
    )
    training.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use pytorch mixed precision training",
    )
    training.add_argument("--seed", default=None, type=int, help="Random seed")
    training.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed training",
    )
    training.add_argument(
        "--weights_init_scale",
        default=0.5,
        type=float,
        help="If set, overwrites value in config.",
    )
    training.add_argument(
        "--hidden_hidden_bias_scale",
        type=float,
        help="If set, overwrites value in config.",
    )
    training.add_argument(
        "--dump_mel_stats",
        action="store_true",
        default=False,
        help="Dump unnormalized mel stats, then stop.",
    )

    optim = parser.add_argument_group("optimization setup")
    optim.add_argument(
        "--num_gpus",
        default=8,
        type=int,
        help="""Number of GPUs to use for training. There are num_gpus processes,
        each running a copy of train.py on one GPU.""",
    )
    optim.add_argument(
        "--global_batch_size",
        default=1024,
        type=int,
        help="Effective batch size across all GPUs after grad accumulation",
    )
    optim.add_argument(
        "--val_batch_size", default=2, type=int, help="Evalution time batch size"
    )
    optim.add_argument("--lr", default=4e-3, type=float, help="Peak learning rate")
    optim.add_argument(
        "--min_lr", default=1e-5, type=float, help="minimum learning rate"
    )
    optim.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight decay for the optimizer",
    )
    optim.add_argument(
        "--grad_accumulation_batches",
        default=8,
        type=int,
        help="Number of batches that must be accumulated for a single model update (step)",
    )
    optim.add_argument(
        "--clip_norm",
        default=1,
        type=float,
        help="If provided, gradients will be clipped above this norm",
    )
    optim.add_argument("--beta1", default=0.9, type=float, help="Beta 1 for optimizer")
    optim.add_argument(
        "--beta2", default=0.999, type=float, help="Beta 2 for optimizer"
    )
    optim.add_argument(
        "--ema",
        type=float,
        default=0.999,
        help="Discount factor for exp averaging of model weights",
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--dali_device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Use DALI pipeline for fast data processing",
    )
    io.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the specified checkpoint or the last saved checkpoint.",
    )
    io.add_argument(
        "--fine_tune",
        action="store_true",
        help="Start training anew from the specified checkpoint.",
    )
    io.add_argument("--ckpt", default=None, type=str, help="Path to a checkpoint")
    io.add_argument(
        "--save_at_the_end",
        action="store_true",
        help="Saves model checkpoint at the end of training",
    )
    io.add_argument(
        "--save_frequency",
        default=None,
        type=int,
        help=(
            "Checkpoint saving frequency in epochs. If 0 (or None), we only possibly save "
            "best and last checkpoints depending on values of --save_best_from and "
            "--save_at_the_end respectively."
        ),
    )
    io.add_argument(
        "--save_best_from",
        default=1,
        type=int,
        help="Epoch on which to begin tracking best checkpoint (dev WER)",
    )
    io.add_argument(
        "--val_frequency",
        default=1,
        type=int,
        help="Number of epochs between evaluations on dev set",
    )
    io.add_argument(
        "--log_frequency",
        default=1,
        type=int,
        help="Number of steps between printing training stats",
    )
    io.add_argument(
        "--prediction_frequency",
        default=None,
        type=int,
        help="Number of steps between printing sample decodings",
    )
    io.add_argument(
        "--model_config",
        default="configs/testing-1023sp_run.yaml",
        type=str,
        required=True,
        help="Path of the model configuration file",
    )
    io.add_argument(
        "--num_buckets",
        type=int,
        default=6,
        help="If provided, samples will be grouped by audio duration, "
        "to this number of buckets, for each bucket, "
        "random samples are batched, and finally "
        "all batches are randomly shuffled",
    )
    io.add_argument(
        "--train_manifests",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the training dataset manifest file"
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--val_manifests",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the evaluation datasets manifest files"
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    io.add_argument(
        "--train_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="One or more paths or globs for the training dataset tar files. "
        "Ignored if --read_from_tar=False. Must be provided if "
        "--read_from_tar=True.",
    )
    io.add_argument(
        "--val_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths (or globs) of the evaluation datasets tar files."
        "Ignored if --read_from_tar=False. Must be provided if "
        "--read_from_tar=True.",
    )
    io.add_argument(
        "--dataset_dir", required=True, type=str, help="Root dir of dataset"
    )
    io.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for logs and checkpoints",
    )
    io.add_argument(
        "--log_file", type=str, default=None, help="Path to save the training logfile."
    )
    io.add_argument(
        "--max_symbol_per_sample",
        type=int,
        default=None,
        help="maximum number of symbols per sample can have during eval",
    )
    io.add_argument(
        "--rsp_seq_len_freq",
        type=int,
        nargs="+",
        default=[99, 0, 1],
        help="""Controls frequency and amount of random state passing

        [100,4,5] means that there will be 100 normal utterances, 4 utterances
        that are 2x longer, and 5 utterances that are 3x longer. Then training
        will loop back to 100 more normal utterances, etc

        This list can be longer: [99,0,0,0,1] means that there will be 99 normal
        utterances and 1 utterance that is 5x longer.

        The model does not always sees exactly 99 normal utterances followed by
        1 longer utterance, since the utterances are in random order.

        "1 utterance that is 5x longer" is implemented by passing the state
        through 5 consecutive utterances. Hence the 5x longer utterance and the
        99 normal utterances use up 104 normal utterances of training data

        To do no state passing, pass [1].

        Experiments suggest a default of [99,0,1]
        """,
    )
    io.add_argument(
        "--rsp_delay",
        type=int,
        default=None,
        help="""Steps of training to do before turning on random state passing. If this
        is None the value defaults to the one set by rnnt_train/common/rsp.py::set_rsp_delay_default.
        See that docstring for more information.
        """,
    )
    io.add_argument(
        "--timestamp",
        default=time.strftime("%Y_%m_%d_%H_%M_%S"),
        type=str,
        help="Timestamp to use for logging",
    )
    io.add_argument(
        "--skip_state_dict_check",
        action="store_true",
        default=False,
        help="Disable checking of model architecture at start of training. This "
        "will result in trained models that are incompatible with downstream inference "
        "server and is intended for experimentation only.",
    )
    return parser


def verify_train_args(args: Namespace) -> Namespace:
    # check data path args
    if not args.read_from_tar:
        assert (
            args.train_manifests is not None
        ), "Must provide train_manifests if not reading from tar"
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"
        assert args.train_tar_files is None and args.val_tar_files is None, (
            "Must not provide tar files if not reading from tar but "
            f"{args.train_tar_files=} and {args.val_tar_files=}.\nDid you mean to "
            "pass --read_from_tar?"
        )
    else:
        assert (
            args.val_tar_files is not None
        ), "Must provide val_tar_files if --read_from_tar=True"
        assert (
            args.train_tar_files is not None
        ), "Must provide train_tar_files if --read_from_tar=True"
    return args


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, "module", model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


def main(args, train_objects):
    args = verify_train_args(args)
    world_size = train_objects.world_size

    assert (
        args.prediction_frequency is None
        or args.prediction_frequency % args.log_frequency == 0
    )
    cfg = train_objects.cfg

    rsp_config_checks(args, cfg)

    assert args.grad_accumulation_batches >= 1

    out_dir = Path(args.output_dir)
    # fail if output dir already contains checkpoints and not resuming or dumping mel stats
    if (
        not args.resume
        and out_dir.exists()
        and any(out_dir.glob("*checkpoint*.pt"))
        and not args.dump_mel_stats
    ):
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
    if args.dump_mel_stats:
        assert (
            not args.num_gpus > 1
        ), "dumping mel stats not supported in multi-gpu mode. Set NUM_GPU=1 to continue"

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

    steps_per_epoch = None
    if not args.read_from_tar:
        # steps per epoch is unknown for tarred data
        steps_per_epoch = len(train_loader) // args.grad_accumulation_batches

    meta = train_objects.training_only.meta
    start_epoch = meta["start_epoch"]
    best_wer = meta["best_wer"]
    step = initial_step = meta["step"]
    if steps_per_epoch is not None:
        start_step = meta["start_epoch"] * steps_per_epoch + 1
        if start_step != step:
            print_once(
                f"WARNING: starting step={start_step} but got step={step} from checkpoint"
            )
            step = start_step
    train_dataset_kw = train_objects.data_objects[TRAIN].dataset_kw
    train_features_kw = train_objects.data_objects[TRAIN].features_kw

    if args.dump_mel_stats:
        # prepare accumulators
        meldim = train_features_kw["n_filt"]
        melsum = torch.zeros(meldim, dtype=torch.float64)
        melss = torch.zeros(meldim, dtype=torch.float64)
        meln = torch.zeros(1, dtype=torch.float64)
        bnum = 0
        # begin accumulating...
        print_once("\n\nDumping mel stats...\n\n")

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
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()

        for batch in train_loader:
            if accumulated_batches == 0:
                train_objects.training_only.adjust_lr(step)
                optimizer_wrapper.zero_grad()
                step_utts = 0
                step_start_time = time.time()
                all_feat_lens = []
                losses = []

            optimizer_wrapper.set_step(step, accumulated_batches)

            # note : these variable names are a bit misleading : 'audio' is already features - rob@myrtle
            audio, audio_lens, txt, txt_lens = batch

            # audio is (batch, meldim, max_len)
            # audio_lens is (batch,)

            if args.dump_mel_stats:
                melsum += torch.sum(audio, (0, 2))
                melss += torch.sum(audio * audio, (0, 2))
                meln += torch.sum(audio_lens)
                bnum += 1
                tot_samples = "unk" if args.read_from_tar else len(train_loader)
                log((epoch, bnum, tot_samples))
                continue

            # if these tensors were computed on cpu then move them to gpu - rob@myrtle
            if args.dali_device == "cpu":
                audio = audio.cuda()
                audio_lens = audio_lens.cuda()
                txt = txt.cuda()
                txt_lens = txt_lens.cuda()

            # now do spec augment / frame stacking - rob@myrtle
            # feats is (seq_len, batch, input_dim)
            feats, feat_lens = train_feat_proc([audio, audio_lens])
            all_feat_lens += feat_lens

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

            if not loss_nan:
                losses.append(loss_item)
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1

            state, rsp_counter = rsp_end_step(state, loss_nan, step, args, rsp_counter)

            # the > 0 condition is a bugfix; absence was causing 1st batch NaNs to enter this code - rob
            if (
                accumulated_batches > 0
                and accumulated_batches % args.grad_accumulation_batches == 0
            ):
                total_norm, tb_per_layer_logs = get_logging_entries(model, scaler)

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
                        preds = decoder.decode(model, feats, feat_lens)
                        wer, pred_utt, ref = greedy_wer(
                            preds, txt, txt_lens, tokenizer.detokenize
                        )
                        print_once(f"  Decoded:   {pred_utt[:90]}")
                        print_once(f"  Reference: {ref[:90]}")
                        wer = {"wer": 100 * wer}
                    else:
                        wer = {}

                    step_time = time.time() - step_start_time
                    # dllogger.log expects a tuple of:
                    # (epoch, <steps in current epoch>, steps_per_epoch)
                    # if steps_per_epoch is None (meaning we are loading data from tar
                    # files and we are on the first epoch) then, the for the sake of
                    # logging we use a tuple of:
                    # (epoch, <steps in current epoch>, 'unk')
                    if steps_per_epoch is not None:
                        step_in_epoch = step % steps_per_epoch or steps_per_epoch
                        log_step_tuple = (epoch, step_in_epoch, steps_per_epoch)
                    else:
                        log_step_tuple = (epoch, step - initial_step + 1, "unk")
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
                            **dict(tb_per_layer_logs),
                        },
                    )

                step_start_time = time.time()

                step += 1
                accumulated_batches = 0
                # end of step

        if args.dump_mel_stats:
            # this should use more stable computations; in practice stdevs are quite large - rob@myrtle
            melmeans = melsum / meln
            melvars = melss / meln - melmeans * melmeans
            # calculated as doubles for precision; convert to float32s for use
            melmeans = melmeans.type(torch.FloatTensor)
            melvars = melvars.type(torch.FloatTensor)
            meln = meln.type(torch.FloatTensor)
            # if these values are going to fail in stream_norm.py, let's find out now
            z = np.zeros_like(melvars)
            np.testing.assert_array_less(
                z, melvars, "\nERROR : All variances should be positive\n"
            )
            # save as PyTorch tensors
            torch.save(melmeans, "/results/melmeans.pt")
            torch.save(melvars, "/results/melvars.pt")
            torch.save(meln, "/results/meln.pt")
            # save as Numpy arrays
            np.save("/results/melmeans.npy", melmeans.numpy())
            np.save("/results/melvars.npy", melvars.numpy())
            np.save("/results/meln.npy", meln.numpy())
            exit()

        epoch_time = time.time() - epoch_start_time
        log(
            (epoch,),
            None,
            "train_avg",
            {"throughput": epoch_utts / epoch_time, "took": epoch_time},
        )

        if steps_per_epoch is None:
            # after running a full epoch we know how many steps there are
            steps_per_epoch = step - initial_step

        if epoch % args.val_frequency == 0:
            wer = evaluate(
                epoch,
                step,
                val_loader,
                val_feat_proc,
                tokenizer.detokenize,
                ema_model,
                loss_fn,
                decoder,
                args,
            )["wer"]

            if wer < best_wer:
                best_wer = wer
                if epoch >= args.save_best_from:
                    checkpointer.save(
                        model,
                        ema_model,
                        optimizer,
                        epoch,
                        step,
                        best_wer,
                        tokenizer_kw,
                        is_best=True,
                    )

        save_this_epoch = bool(args.save_frequency) and epoch % args.save_frequency == 0
        if save_this_epoch:
            checkpointer.save(
                model, ema_model, optimizer, epoch, step, best_wer, tokenizer_kw
            )

    log((), None, "train_avg", {"throughput": epoch_utts / epoch_time})

    if epoch == args.epochs:
        evaluate(
            epoch,
            step,
            val_loader,
            val_feat_proc,
            tokenizer.detokenize,
            ema_model,
            loss_fn,
            decoder,
            args,
        )

    flush_log()
    if args.save_at_the_end:
        checkpointer.save(
            model, ema_model, optimizer, epoch, step, best_wer, tokenizer_kw
        )


if __name__ == "__main__":
    parser = train_arg_parser()
    args = parser.parse_args()
    train_objects = TrainSetup().run(args)
    main(args, train_objects)
