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


# valCPU.py is derived from val.py
# rob@myrtle, May 2022


import argparse
import copy
import os
import random
import time

import torch
import numpy as np

from common import helpers
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from common.data import features
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.tb_dllogger import flush_log, init_log, log
from common.stream_norm import StreamNorm
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.model import RNNT

from mlperf import logging


def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training/validation setup')
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use mixed precision')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed processing')
    training.add_argument('--streaming_normalization', action='store_true', default=False,
                          help='Use streaming normalization instead of DALI normalization.')
    training.add_argument('--reset_stream_stats', action='store_true', default=False,
                          help='Reset streaming normalization statistics for every sentence.')
    training.add_argument('--alpha', default=0.001, type=float,
                          help='Streaming normalization decay coefficient, 0<=alpha<1')
    training.add_argument('--dump_nth', default=None, type=int,
                          help='Dump dither-off tensors from the nth batch to /results/ and exit')
    training.add_argument('--dump_preds', action='store_true', default=False,
                          help='Dump text predictions to /results/preds.txt')
    
    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--val_batch_size', default=1, type=int,
                       help='Evalution time batch size')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to the checkpoint to use')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None,
                    help='maximum number of symbols per sample can have during eval')
    return parser.parse_args()


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
             ema_model, greedy_decoder, stream_norm, args):

    ema_model.eval()

    dumptype = None
    if args.dump_nth != None:
        if stream_norm:
            dumptype = 'stream'
        else:
            dumptype = 'dali'

    if args.reset_stream_stats:
        training_means = torch.load('/results/melmeans.pt')
        training_vars  = torch.load('/results/melvars.pt')
    
    start_time = time.time()
    agg = {'preds': [], 'txts': [], 'idx': []}
    logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num=epoch))
    for i, batch in enumerate(val_loader):
        print(f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}', end='\r')
        
        # note : these variable names are a bit misleading : 'audio' is already features - rob@myrtle
        audio, audio_lens, txt, txt_lens = batch

        if args.dump_nth != None and i == args.dump_nth and stream_norm:
            np.save(f'/results/logmels{i}.npy', audio.numpy())

        if stream_norm:
            # Then the audio tensor was not normalized by DALI and must be normalized here.
            # The Rust Inference Server inits each new Channel with the stream-norm training stats.
            # The args.reset_stream_stats option acts similarly for each new utterance in the manifest.
            # The Python system can then match the Rust system using a new Channel for each utterance.
            if args.reset_stream_stats:
                stream_norm.mel_means = training_means.clone()
                stream_norm.mel_vars  = training_vars.clone()
            # The stream_norm class normalizes over time using an exponential moving average.
            # The stats held in the stream_norm class are updated each frame, effectively adapting to
            # the current speaker.  audio is (batch, meldim, time) and batch is enforced to be 1 below.
            for j in range(audio.shape[2]):
                audio[0, :, j] = stream_norm.normalize(audio[0, :, j])

        # By default the stream-norm stats are updated by every utterance in the manifest.
        # This continue statement therefore follows the update (above).
        if args.dump_nth != None and i < args.dump_nth:
            continue
        
        if args.dump_nth != None:
            np.save(f'/results/{dumptype}norm{i}.npy', audio.numpy())

        # now do frame stacking - rob@myrtle
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        if args.dump_nth != None:
            np.save(f'/results/{dumptype}stack{i}.npy', feats.numpy())
        
        pred = greedy_decoder.decode(ema_model, feats, feat_lens, dumptype, i)
        
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts']  += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

        if args.dump_nth != None:
            with open(f'/results/{dumptype}preds{i}.txt', 'w') as f:
                for line in agg['preds']:
                    f.write(str(line))
                    f.write('\n')
            with open(f'/results/txts{i}.txt', 'w') as f:
                for line in agg['txts']:
                    f.write(str(line))
                    f.write('\n')
            exit()
    
    wer, loss = process_evaluation_epoch(agg)

    logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch))
    log((epoch,), step, 'dev_ema', {'loss': loss, 'wer': 100.0 * wer, 'took': time.time() - start_time})

    if args.dump_preds:
        with open(f'/results/preds.txt', 'w') as f:
            for line in agg['preds']:
                f.write(str(line))
                f.write('\n')

    return wer


def main():
    logging.configure_logger('RNNT')
    logging.log_start(logging.constants.INIT_START)

    args = parse_args()

    if args.streaming_normalization:
        if args.val_batch_size != 1:
            print("streaming normalization requires val_batch_size of 1")
            exit()

    if args.dump_nth != None:
        if args.val_batch_size != 1:
            print("dump_nth requires val_batch_size of 1 (to prevent logp overwrites)")
            exit()

    # Set PyTorch to run on one CPU thread to ensure deterministic PyTorch output.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

    init_log(args)

    cfg = config.load(args.model_config)
    
    logging.log_end(logging.constants.INIT_STOP)
    logging.log_start(logging.constants.RUN_START)

    print_once('Setting up datasets...')
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, 'val')

    if args.dump_nth != None:
        val_features_kw['dither'] = 0.0
    
    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    class PermuteAudio(torch.nn.Module):
        def forward(self, x):
            return (x[0].permute(2, 0, 1), *x[1:])

    val_augmentations = torch.nn.Sequential(
        val_specaugm_kw and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        PermuteAudio(),
    )

    val_loader = DaliDataLoader(gpu_id=None, # Use None as a device_id to run DALI without CUDA
                                dataset_path=args.dataset_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=args.val_manifests,
                                batch_size=args.val_batch_size,
                                sampler=dali_sampler.SimpleSampler(),
                                pipeline_type="val",
                                normalize=not args.streaming_normalization,
                                num_cpu_threads=1,
                                device_type="cpu",
                                tokenizer=tokenizer)
    
    val_feat_proc   = val_augmentations

    logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    # set up the model
    rnnt_config = config.rnnt(cfg)
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    blank_idx = tokenizer.num_labels
    logging.log_event(logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=args.max_symbol_per_sample)
    greedy_decoder = RNNTGreedyDecoder( blank_idx=blank_idx,
                                        max_symbol_per_sample=args.max_symbol_per_sample)

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    ema_model = copy.deepcopy(model)

    # A checkpoint should always be specified
    assert args.ckpt is not None

    # setup checkpointer
    checkpointer = Checkpointer(args.output_dir, 'RNN-T', [], args.amp)

    # load checkpoint (modified to not need optimizer / meta args)
    checkpointer.load(args.ckpt, model, ema_model)

    # setup streaming normalizer
    if args.streaming_normalization:
        melmeans    = torch.load('/results/melmeans.pt')
        melvars     = torch.load('/results/melvars.pt')
        stream_norm = StreamNorm(args.alpha, melmeans, melvars)
        print_once(f"Using streaming normalization, alpha={args.alpha}, "
                   f"reset_stream_stats={args.reset_stream_stats}")
    else:
        stream_norm = None
        
    epoch = 1
    step  = None # Switches off logging of val data results to TensorBoard

    wer = evaluate(epoch, step, val_loader, val_feat_proc,
                   tokenizer.detokenize, ema_model,
                   greedy_decoder, stream_norm, args)

    flush_log()
    print_once(f'\nWord Error Rate: {wer*100.0:5.3f}% on {" ".join(args.val_manifests)}\n')


if __name__ == "__main__":
    main()
