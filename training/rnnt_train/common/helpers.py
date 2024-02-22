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

import glob
import os
import re
from collections import OrderedDict

import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple

from rnnt_train.common.metrics import word_error_rate


def __rnnt_decoder_predictions_tensor(tensor, detokenize):
    """
    Takes output of greedy rnnt decoder and converts to strings.
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    return [detokenize(pred) for pred in tensor]


@beartype
def is_rank_zero() -> bool:
    return get_rank_or_zero() == 0


@beartype
def get_rank_or_zero() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def print_once(msg):
    if is_rank_zero():
        print(msg)


def calculate_wer(preds, tgt, tgt_lens, detokenize, standardize_wer):
    """
    Calculates WER and returns an example hypothesis and reference.
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    wer, _, _ = word_error_rate(hypotheses, references, standardize=standardize_wer)
    return wer, hypotheses[0], references[0]


def gather_losses(losses_list):
    return [torch.mean(torch.stack(losses_list))]


def gather_predictions(predictions_list, detokenize):
    rnnt_predictions = (
        __rnnt_decoder_predictions_tensor(prediction, detokenize)
        for prediction in predictions_list
    )

    return [prediction for batch in rnnt_predictions for prediction in batch]


def gather_transcripts(transcript_list, transcript_len_list, detokenize):
    return [
        detokenize(t[:l].long().cpu().numpy().tolist())
        for txt, lens in zip(transcript_list, transcript_len_list)
        for t, l in zip(txt, lens)
    ]


@beartype
def process_evaluation_epoch(
    aggregates: Dict[str, list], standardize_wer: bool
) -> Tuple[float, Optional[float]]:
    """
    Processes results from each worker at the end of evaluation and combine to final result

    Aggregates will be updated in-place to have timestamps from all processes.

    Args:
        aggregates: dictionary containing information of entire evaluation
        standardize_wer: whether to apply Whisper normalizatio rules to
        the transcripts
    Return:
        wer: final word error rate
        loss: final loss
    """
    if "losses" in aggregates:
        eloss = torch.mean(torch.stack(aggregates["losses"])).item()
    else:
        eloss = None

    hypotheses = aggregates["preds"]
    references = aggregates["txts"]

    wer, scores, num_words = word_error_rate(
        hypotheses, references, standardize=standardize_wer
    )
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        wer = scores * 1.0 / num_words

        gathered_results = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        # Gather aggregates dictionary across all workers
        dist.gather_object(aggregates, gathered_results, dst=0)

        # If rank 0 process, combine timestamp data from all processes
        if dist.get_rank() == 0:
            lst_seq_time = [
                timestamp
                for results in gathered_results
                for timestamp in results["timestamps"]
            ]
            aggregates["timestamps"] = lst_seq_time
    return wer, eloss


def num_weights(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def unwrap_ddp(model):
    return getattr(model, "module", model)


class Checkpointer(object):
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name

        tracked = [
            (int(re.search(r"epoch(\d+)_", f).group(1)), f)
            for f in glob.glob(f"{save_dir}/{self.model_name}_epoch*_checkpoint.pt")
        ]
        tracked = sorted(tracked, key=lambda t: t[0])
        self.tracked = OrderedDict(tracked)

    def save(
        self,
        model,
        ema_model,
        optimizer,
        epoch,
        step,
        best_wer,
        tokenizer_kw,
        is_best=False,
    ):
        """Saves model checkpoint for inference/resuming training.

        Args:
            model: the model, optionally wrapped by DistributedDataParallel
            ema_model: model with averaged weights, can be None
            optimizer: optimizer
            epoch (int): epoch during which the model is saved
            step (int): number of steps since beginning of training
            best_wer (float): lowest recorded WER on the dev set
            tokenizer_kw: Dictionary of details about the tokenizer,
                          including the list of characters (key "labels"),
                          and (if one is used) the path to the sentencepiece model
                          (key "sentpiece_model")
            is_best (bool, optional): set name of checkpoint to 'best'
                and overwrite the previous one
        """
        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        if is_best:
            fpath = os.path.join(self.save_dir, f"{self.model_name}_best_checkpoint.pt")
        else:
            fpath = os.path.join(
                self.save_dir, f"{self.model_name}_epoch{epoch}_checkpoint.pt"
            )

        # Checkpoint already saved
        if not is_best and epoch in self.tracked:
            print_once(f"WARNING: Overwriting previous checkpoint {fpath}")

        state = {
            "epoch": epoch,
            "step": step,
            "best_wer": best_wer,
            "state_dict": unwrap_ddp(model).state_dict(),
            "ema_state_dict": unwrap_ddp(ema_model).state_dict()
            if ema_model is not None
            else None,
            "optimizer": optimizer.state_dict(),
            "tokenizer_kw": tokenizer_kw,
        }

        print_once(f"Saving {fpath}...")
        torch.save(state, fpath)

        if not is_best:
            self.tracked[epoch] = fpath

    def last_checkpoint(self):
        tracked = list(self.tracked.values())

        if len(tracked) >= 1:
            try:
                torch.load(tracked[-1], map_location="cpu")
                return tracked[-1]
            except Exception:
                print_once(f"Last checkpoint {tracked[-1]} appears corrupted.")

        elif len(tracked) >= 2:
            return tracked[-2]
        else:
            return None

    def load(self, fpath, model, ema_model, optimizer=None, meta=None):
        """Modified to support Test data evaluations which don't need optimizers/meta"""

        print_once(f"Loading model from {fpath}")
        checkpoint = torch.load(fpath, map_location="cpu")

        state_dict = checkpoint["state_dict"]
        unwrap_ddp(model).load_state_dict(state_dict, strict=True)

        if ema_model is not None:
            if checkpoint.get("ema_state_dict") is not None:
                key = "ema_state_dict"
            else:
                key = "state_dict"
                print_once("WARNING: EMA weights not found in the checkpoint.")
                print_once("WARNING: Initializing EMA model with regular params.")
            state_dict = checkpoint[key]
            unwrap_ddp(ema_model).load_state_dict(state_dict, strict=True)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if meta is not None:
            meta["start_epoch"] = checkpoint.get("epoch")
            meta["best_wer"] = checkpoint.get("best_wer", meta["best_wer"])
            meta["step"] = checkpoint.get("step", meta["step"])

        return checkpoint.get("tokenizer_kw")
