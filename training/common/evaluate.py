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

# The evaluate() function was originally in train.py - rob@myrtle

import time
import torch
from common import helpers
from common.helpers import process_evaluation_epoch
from common.tb_dllogger import log
from mlperf import logging

@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
             ema_model, loss_fn, greedy_decoder, args):

    ema_model.eval()

    start_time = time.time()
    agg = {'losses': [], 'preds': [], 'txts': [], 'idx': []}
    logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num=epoch))
    for i, batch in enumerate(val_loader):
        print(f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}', end='\r')

        # note : these variable names are a bit misleading : 'audio' is already features - rob@myrtle
        audio, audio_lens, txt, txt_lens = batch

        # if these tensors were computed on cpu then move them to gpu - rob@myrtle
        if args.dali_device == "cpu":
            audio      = audio.cuda()
            audio_lens = audio_lens.cuda()
            txt        = txt.cuda()
            txt_lens   = txt_lens.cuda()

        # now do frame stacking - rob@myrtle
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        # note : more misleading variable names : 'log_prob*' are actually logits - rob@myrtle
        log_probs, log_prob_lens = ema_model(feats, feat_lens, txt, txt_lens)
        # batch_offset and max_f_len parameters are required for the apex transducer loss
        batch_offset = torch.cumsum(feat_lens*(txt_lens+1), dim=0)
        max_f_len = max(feat_lens)
        loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                       log_prob_lens, txt, txt_lens,
                       batch_offset=batch_offset,
                       max_f_len=max_f_len,
                       )

        pred = greedy_decoder.decode(ema_model, feats, feat_lens)

        agg['losses'] += helpers.gather_losses([loss.cpu()])
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

    wer, loss = process_evaluation_epoch(agg)

    logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch))

    log((epoch,), step, 'dev_ema', {'loss': loss, 'wer': 100.0 * wer, 'took': time.time() - start_time})
    ema_model.train()
    return wer

