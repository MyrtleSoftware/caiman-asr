import glob
import math
import os
import re
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.export.hardware_ckpt import main as make_hw_ckpt
from caiman_asr_train.export.model_schema import get_schema, return_schemas
from caiman_asr_train.train_utils.distributed import print_once, unwrap_ddp


@beartype
class Checkpointer(object):
    def __init__(self, save_dir, model_name, allow_partial_load: bool = False):
        self.save_dir = save_dir
        self.model_name = model_name
        self.allow_partial_load = allow_partial_load

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
        logmel_norm_weight: float,
        config_path: str,
        is_best: bool = False,
        is_last: bool = False,
        filepath: Optional[str] = None,
    ) -> None:
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
            is_last (bool, optional): set name of checkpoint to 'last'
                                      to save the final step checkpoint.
        """
        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        if filepath:
            fpath = filepath
        elif is_best:
            fpath = os.path.join(self.save_dir, f"{self.model_name}_best_checkpoint.pt")
        elif is_last:
            fpath = os.path.join(self.save_dir, f"{self.model_name}_last_checkpoint.pt")
        else:
            fpath = os.path.join(
                self.save_dir,
                f"{self.model_name}_step{step}_checkpoint.pt",
            )

        # Checkpoint already saved
        if not (is_best or is_last) and step in self.tracked:
            print_once(f"WARNING: Overwriting previous checkpoint {fpath}")

        model_state_dict = unwrap_ddp(model).state_dict()
        state = {
            "epoch": epoch,
            "step": step,
            "best_wer": best_wer,
            "state_dict": model_state_dict,
            "ema_state_dict": (
                unwrap_ddp(ema_model).state_dict() if ema_model is not None else None
            ),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "tokenizer_kw": tokenizer_kw,
            "logmel_norm_weight": logmel_norm_weight,
        }

        print_once(f"Saving {fpath}...")
        torch.save(state, fpath, pickle_protocol=5)

        if is_best or is_last:
            self.save_hardware_checkpoint(
                fpath, logmel_norm_weight, config_path, model_state_dict
            )

        if not is_best and not is_last:
            self.tracked[step] = fpath

    def save_hardware_checkpoint(
        self,
        fpath: str,
        logmel_norm_weight: float,
        config: str,
        model_state_dict: dict,
    ) -> None:
        if get_schema(model_state_dict) not in return_schemas():
            print_once(
                "Not saving hardware checkpoint as model is not supported on FPGA"
            )
            return
        if not math.isclose(logmel_norm_weight, 1.0):
            print_once(
                f"Not saving hardware checkpoint as {logmel_norm_weight=} is not yet 1.0"
            )
            return

        output_ckpt = Path(fpath).with_suffix(".hw.pt")
        hw_ckpt_args = Namespace(
            ckpt=fpath,
            config=config,
            output_ckpt=output_ckpt,
            override_ngram_path=None,
            skip_ngram=False,
        )
        make_hw_ckpt(hw_ckpt_args)
        print_once(f"Saved hardware checkpoint to {output_ckpt}")

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

    def _load(self, model, state_dict, quiet=False):
        """
        Load a model from a state dict using strict=False, and print warnings
        about missing or unexpected keys. Throws an error if no keys are loaded.
        """

        missing, unexpected = model.load_state_dict(
            state_dict, strict=not self.allow_partial_load
        )

        maybe_print_once = (lambda _: None) if quiet else print_once  # noqa: E731

        if not unexpected and not missing:
            maybe_print_once("Loaded all parameters from checkpoint.")
            return

        if loaded_keys := set(k for k in state_dict.keys() if k not in unexpected):
            maybe_print_once("WARNING: Checkpoint is partially loaded, using:")
            for key in loaded_keys:
                maybe_print_once(f"  {key}")
        else:
            # Always print as we will raise an error
            print_once("In checkpoint:")
            for key in state_dict.keys():
                print_once(f"  {key}")

            print_once("Model expects:")
            for key in model.state_dict().keys():
                print_once(f"  {key}")

            raise ValueError("No keys loaded from the checkpoint.")

        if quiet:
            return

        if unexpected:
            print_once("WARNING: Unused keys in the checkpoint:")
            for key in unexpected:
                print_once(f"  {key}")

        if missing:
            print_once("WARNING: Missing keys in the checkpoint:")
            for key in missing:
                print_once(f"  {key}")

    def load(self, fpath, model, ema_model, optimizer=None, meta=None):
        """Modified to support Test data evaluations which don't need optimizers/meta"""

        print_once(f"Loading model from {fpath}")
        checkpoint = torch.load(fpath, map_location="cpu")

        self._load(unwrap_ddp(model), checkpoint["state_dict"])

        if ema_model is not None:
            if checkpoint.get("ema_state_dict") is not None:
                key = "ema_state_dict"
            else:
                key = "state_dict"
                print_once("WARNING: EMA weights not found in the checkpoint.")
                print_once("WARNING: Initializing EMA model with regular params.")

            self._load(unwrap_ddp(ema_model), checkpoint[key], quiet=True)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if meta is not None:
            meta["start_epoch"] = checkpoint.get("epoch")
            meta["best_wer"] = checkpoint.get("best_wer", meta["best_wer"])
            meta["step"] = checkpoint.get("step", meta["step"])

        return checkpoint.get("tokenizer_kw")
