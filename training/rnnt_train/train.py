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

# modified by rob@myrtle

import argparse
import copy
import json
import os
import random
import shutil
import time
from argparse import Namespace
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.distributed as dist
from apex.optimizers import FusedLAMB
from torch.nn.parallel import DistributedDataParallel

from rnnt_train.common.data import features
from rnnt_train.common.data.build_dataloader import build_dali_loader
from rnnt_train.common.data.dali import extract_melmat as dali_melmat
from rnnt_train.common.data.dali import sampler as dali_sampler
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.evaluate import evaluate
from rnnt_train.common.helpers import Checkpointer, greedy_wer, num_weights, print_once
from rnnt_train.common.logging_layers import get_logging_entries
from rnnt_train.common.optimizers import lr_policy
from rnnt_train.common.seed import set_seed
from rnnt_train.common.tb_dllogger import flush_log, init_log, log
from rnnt_train.rnnt import config
from rnnt_train.rnnt.decoder import RNNTGreedyDecoder
from rnnt_train.rnnt.loss import apexTransducerLoss
from rnnt_train.rnnt.model import RNNT


def parse_args() -> Namespace:
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
        default=10880,
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
        "--epochs_this_job",
        default=0,
        type=int,
        help=(
            "Run for a number of epochs with no effect on the lr schedule."
            "Useful for re-starting the training."
        ),
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
        "--dump_melmat",
        type=str,
        choices=["none", "rosa", "dali"],
        default="none",
        help="Dump fft-to-mel transform matrices.",
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
        default=1e-3,
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
        "--max_duration", type=float, help="Discard samples longer than max_duration"
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
        "--timestamp",
        default=time.strftime("%Y_%m_%d_%H_%M_%S"),
        type=str,
        help="Timestamp to use for logging",
    )
    args = parser.parse_args()

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


def main():
    args = parse_args()

    assert torch.cuda.is_available()
    assert (
        args.prediction_frequency is None
        or args.prediction_frequency % args.log_frequency == 0
    )

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed training
    multi_gpu = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        print_once(f"Distributed training with {world_size} GPUs\n")
    else:
        world_size = 1

    if args.seed is not None:
        np_rng = set_seed(args.seed, args.local_rank)
        # np_rng is used for buckets generation, and needs the same seed on every worker

    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    assert args.grad_accumulation_batches >= 1

    # Effective batch size per GPU after grad accumulation
    accum_batch_size = int(args.global_batch_size / args.num_gpus)

    assert (
        accum_batch_size % args.grad_accumulation_batches == 0
    ), f"{accum_batch_size=} % {args.grad_accumulation_batches=} != 0"
    batch_size = accum_batch_size // args.grad_accumulation_batches

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

    init_log(args)

    if multi_gpu:
        torch.distributed.barrier()

    print_once("Setting up datasets...")
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
    ) = config.input(cfg, "train")
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, "val")

    # if mel stats are being collected these are for use in inference streaming normalization
    # the stats should therefore reflect the processing that will be used in inference
    if args.dump_mel_stats:
        train_dataset_kw["speed_perturbation"] = None
        train_dataset_kw["trim_silence"] = False

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    train_augmentations = torch.nn.Sequential(
        train_specaugm_kw
        and features.SpecAugment(optim_level=args.amp, **train_specaugm_kw)
        or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **train_splicing_kw),
        features.PermuteAudio(),
    )
    val_augmentations = torch.nn.Sequential(
        val_specaugm_kw
        and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw)
        or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        features.PermuteAudio(),
    )

    if args.read_from_tar is None:
        train_sampler = None
    elif args.num_buckets is not None:
        train_sampler = dali_sampler.BucketingSampler(
            args.num_buckets, batch_size, world_size, args.epochs, np_rng
        )
    else:
        train_sampler = dali_sampler.SimpleSampler()

    train_loader = build_dali_loader(
        args,
        "train",
        batch_size=batch_size,
        dataset_kw=train_dataset_kw,
        features_kw=train_features_kw,
        train_sampler=train_sampler,
        tokenizer=tokenizer,
    )

    val_loader = build_dali_loader(
        args,
        "val",
        batch_size=args.val_batch_size,
        dataset_kw=val_dataset_kw,
        features_kw=val_features_kw,
        tokenizer=tokenizer,
    )

    train_feat_proc = train_augmentations
    val_feat_proc = val_augmentations

    train_feat_proc.cuda()
    val_feat_proc.cuda()

    steps_per_epoch = None
    if not args.read_from_tar:
        # steps per epoch is unknown for tarred data
        steps_per_epoch = len(train_loader) // args.grad_accumulation_batches

    # set up the RNNT model
    rnnt_config = config.rnnt(cfg)
    if args.weights_init_scale is not None:
        rnnt_config["weights_init_scale"] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config["hidden_hidden_bias_scale"] = args.hidden_hidden_bias_scale
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.cuda()
    blank_idx = tokenizer.num_labels
    loss_fn = apexTransducerLoss(blank_idx=blank_idx, packed_input=False)
    greedy_decoder = RNNTGreedyDecoder(
        blank_idx=blank_idx, max_symbol_per_sample=args.max_symbol_per_sample
    )

    print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

    opt_eps = 1e-9

    # optimization
    kw = {
        "params": model.param_groups(args.lr),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    initial_lrs = [group["lr"] for group in kw["params"]]

    print_once(f"Starting with LRs: {initial_lrs}")
    optimizer = FusedLAMB(
        betas=(args.beta1, args.beta2), eps=opt_eps, max_grad_norm=args.clip_norm, **kw
    )

    adjust_lr = lambda step: lr_policy(
        optimizer,
        initial_lrs,
        args.min_lr,
        step,
        args.warmup_steps,
        args.hold_steps,
        args.half_life_steps,
    )
    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None

    if multi_gpu:
        model = DistributedDataParallel(model)

    # load checkpoint
    meta = {"best_wer": 10**6, "start_epoch": 0, "step": 1}
    checkpointer = Checkpointer(args.output_dir, "RNN-T")

    # we cannot both resume and fine_tune
    if args.resume or args.fine_tune:
        assert args.resume ^ args.fine_tune, "cannot both resume and fine_tune"

    # when resuming, a specified checkpoint over-rules any last checkpoint we might find
    # when resuming, we keep optimizer state and meta info
    if args.resume:
        args.ckpt = args.ckpt or checkpointer.last_checkpoint()
        assert args.ckpt is not None, "no checkpoint to resume from"
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)

    # fine-tuning involves taking a trained model and re-training it after some change in model / data
    # when fine tuning, we expect a specified checkpoint
    # when fine-tuning, we do not keep optimizer state and meta info
    if args.fine_tune:
        assert args.ckpt is not None, "no checkpoint to fine_tune from"
        checkpointer.load(args.ckpt, model, ema_model)

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

    if args.dump_melmat != "none":
        # Prepare & dump the matrix that converts FFT frequency bin values into mel bin values.
        srate = train_dataset_kw["sample_rate"]
        n_fft = train_features_kw["n_fft"]
        meldim = train_features_kw["n_filt"]
        # It turns out that the matrix implicit in DALI is very similar to that provided by librosa.
        if args.dump_melmat == "rosa":
            melmat = librosa.filters.mel(sr=srate, n_fft=n_fft, n_mels=meldim)
            fname = f"/results/rosa_melmat{meldim}x{n_fft//2 + 1}.npy"
            sname = f"/results/rosa_sparse{meldim}x{n_fft//2 + 1}.npy"
        # Or we can extract the matrix implicit in the DALI algorithm exactly.
        if args.dump_melmat == "dali":
            melmat = dali_melmat.extract(srate, n_fft, meldim)
            fname = f"/results/dali_melmat{meldim}x{n_fft//2 + 1}.npy"
            sname = f"/results/dali_sparse{meldim}x{n_fft//2 + 1}.npy"
        # Save the full matrix to /results/
        np.save(fname, melmat)
        # The full matrix is usually very sparse; compute a sparse representation here.
        sl = []
        for r, row in enumerate(melmat):
            for c, ent in enumerate(row):
                if ent != 0.0:
                    sl.append([r, c, ent])
        sa = np.array(sl, np.float32)
        # Save the sparse matrix to /results/
        np.save(sname, sa)

    if args.dump_mel_stats:
        # prepare accumulators
        meldim = train_features_kw["n_filt"]
        melsum = torch.zeros(meldim, dtype=torch.float64)
        melss = torch.zeros(meldim, dtype=torch.float64)
        meln = torch.zeros(1, dtype=torch.float64)
        bnum = 0
        # begin accumulating...
        print_once("\n\nDumping mel stats...\n\n")

    # training loop
    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()

        for batch in train_loader:
            if accumulated_batches == 0:
                adjust_lr(step)
                optimizer.zero_grad()
                step_utts = 0
                step_start_time = time.time()
                all_feat_lens = []
                losses = []

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

            # parameters used for the APEX transducer loss function
            batch_offset = torch.cumsum(feat_lens * (txt_lens + 1), dim=0)
            max_f_len = max(feat_lens)

            if args.amp:
                # use autocast from pytorch AMP
                with torch.cuda.amp.autocast():
                    # note : more misleading variable names : 'log_prob*' are actually logits - rob@myrtle
                    # log_probs are cast to float16 w/ autocast.
                    log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
                    loss = loss_fn(
                        log_probs[:, : log_prob_lens.max().item()],
                        log_prob_lens,
                        txt,
                        txt_lens,
                        batch_offset=batch_offset,
                        max_f_len=max_f_len,
                    )
                    loss /= args.grad_accumulation_batches
            else:
                # log_probs are float32
                log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
                loss = loss_fn(
                    log_probs[:, : log_prob_lens.max().item()],
                    log_prob_lens,
                    txt,
                    txt_lens,
                    batch_offset=batch_offset,
                    max_f_len=max_f_len,
                )
                loss /= args.grad_accumulation_batches

            del log_probs, log_prob_lens

            if torch.isnan(loss).any():
                print_once(f"WARNING: loss is NaN; skipping update")
                del loss
            else:
                if args.amp:
                    # scale losses w/ pytorch AMP to prevent underflowing before backward pass
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                losses.append(loss.item())
                del loss
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1

            # the > 0 condition is a bugfix; absence was causing 1st batch NaNs to enter this code - rob
            if (
                accumulated_batches > 0
                and accumulated_batches % args.grad_accumulation_batches == 0
            ):
                total_norm, tb_per_layer_logs = get_logging_entries(model, scaler)

                if args.amp:
                    # pyTorch AMP step function unscales the gradients
                    # if these gradients do not contain infs or NaNs, optimizer.step() is then called
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # when not using AMP we must test for inf / NaN gradients ourselves
                    if np.isfinite(total_norm):
                        optimizer.step()

                apply_ema(model, ema_model, args.ema)

                if step % args.log_frequency == 0:
                    if (
                        args.prediction_frequency is None
                        or step % args.prediction_frequency == 0
                    ):
                        preds = greedy_decoder.decode(model, feats, feat_lens)
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
                            "lrate": optimizer.param_groups[0]["lr"],
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
                greedy_decoder,
                args,
            )

            if wer < best_wer and epoch >= args.save_best_from:
                checkpointer.save(
                    model, ema_model, optimizer, epoch, step, best_wer, is_best=True
                )
                best_wer = wer

        save_this_epoch = bool(args.save_frequency) and epoch % args.save_frequency == 0
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f"Finished after {args.epochs_this_job} epochs.")
            break
        # end of epoch

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
            greedy_decoder,
            args,
        )

    flush_log()
    if args.save_at_the_end:
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    main()
