#!/usr/bin/env python

# Copyright (c) 2022 Myrtle.ai
# rob

import torch
import argparse

parser = argparse.ArgumentParser(description='Gather training results into a hardware checkpoint')
parser.add_argument('--fine_tuned_ckpt', type=str, default='/results/RNN-T_best_checkpoint.pt',
                    help='fine-tuned checkpoint file')
parser.add_argument('--config', type=str, default='configs/baseline_v3-1023sp.yaml',
                    help='config file')
parser.add_argument('--melmeans', type=str, default='/results/melmeans.pt',
                    help='mel means file')
parser.add_argument('--melvars', type=str, default='/results/melvars.pt',
                    help='mel vars file')
parser.add_argument('--melalpha', type=float, default=0.001,
                    help='streaming normalization time constant')
parser.add_argument('--output_ckpt', type=str, default='/results/hardware_ckpt.pt',
                    help='name of the output hardware checkpoint file')
args = parser.parse_args()

# the output hardware checkpoint is a dictionary
hardcp  = dict()

# load the fine-tuned training checkpoint and copy over the required fields
traincp = torch.load(args.fine_tuned_ckpt, map_location="cpu")

# It is useful to be able to sanity-check these hardware checkpoints using val.py on Python.
# So ema_state_dict is renamed to state_dict in the hardware checkpoint which val.py will load with warnings. 
hardcp['state_dict']     = traincp['ema_state_dict']
hardcp['epoch']          = traincp['epoch']
hardcp['step']           = traincp['step']
hardcp['best_wer']       = traincp['best_wer']

# load the mel stats and store them in the hardware checkpoint
hardcp['melmeans'] = torch.load(args.melmeans, map_location="cpu")
hardcp['melvars']  = torch.load(args.melvars,  map_location="cpu")
hardcp['melalpha'] = args.melalpha

# extract the sentencepiece model filename from the config file
spmfn = ''
with open(args.config, 'r') as f:
  for l in f:
    ll = l.split()
    if len(ll) > 0 and ll[0] == 'sentpiece_model:':
      spmfn = ll[1]
      break

assert(spmfn)

# read the sentencepiece model file into a bytes object
with open(spmfn, 'rb') as f:
  spmb = f.read()

# store the bytes object in the hardware checkpoint
hardcp['sentpiece_model'] = spmb

# save the hardware checkpoint to disk
torch.save(hardcp, args.output_ckpt)

