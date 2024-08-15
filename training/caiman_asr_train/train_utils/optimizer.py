from argparse import Namespace

import numpy as np
from beartype import beartype
from beartype.typing import Optional
from torch.cuda.amp import GradScaler

from caiman_asr_train.train_utils.distributed import print_once


@beartype
class OptimizerWrapper:
    """Wrapper to control the optimizer and AMP scaling during training."""

    def __init__(
        self,
        args: Namespace,
        optimizer,
        scaler: Optional[GradScaler],
        lower_bound: Optional[float] = None,
    ):
        self.args = args
        self.optimizer = optimizer
        self.scaler = scaler
        self.scale = None
        self.lower_bound = lower_bound

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, total_norm: float) -> None:
        if not self.args.no_amp:
            self.do_scaler_step()
            self.scaler.update(self.scale)

            if self.lower_bound is not None:
                scale = self.scaler.get_scale()

                if scale < self.lower_bound:
                    print_once("WARNING: Overriding the grad scaler")
                    self.scale = self.lower_bound
                else:
                    self.scale = None

        else:
            # when not using AMP test for inf / NaN gradients
            if np.isfinite(total_norm):
                self.optimizer.step()

    def do_scaler_step(self) -> None:
        # pyTorch AMP step function unscales the gradients
        # if these gradients do not contain infs or NaNs, optimizer.step() is then called
        self.scaler.step(self.optimizer)

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
