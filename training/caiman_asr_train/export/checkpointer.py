import glob
import os
import re
from collections import OrderedDict

import torch
import torch.distributed as dist

from caiman_asr_train.train_utils.distributed import print_once, unwrap_ddp


class Checkpointer(object):
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name

        tracked = [
            (int(re.search(r"step(\d+)_", f).group(1)), f)
            for f in glob.glob(f"{save_dir}/{self.model_name}_step*_checkpoint.pt")
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
        logmel_norm_weight,
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
            logmel_norm_weight (float): current weight for logmel normalization.
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
                self.save_dir,
                f"{self.model_name}_step{step}_checkpoint.pt",
            )

        # Checkpoint already saved
        if not is_best and step in self.tracked:
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
            "logmel_norm_weight": logmel_norm_weight,
        }

        print_once(f"Saving {fpath}...")
        torch.save(state, fpath)

        if not is_best:
            self.tracked[step] = fpath

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
