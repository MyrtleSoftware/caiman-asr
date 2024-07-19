from argparse import Namespace

import numpy as np
from beartype import beartype
from beartype.typing import Optional
from torch.cuda.amp import GradScaler


@beartype
class OptimizerWrapper:
    """Wrapper to control the optimizer and AMP scaling during training."""

    def __init__(
        self,
        args: Namespace,
        optimizer,
        scaler: Optional[GradScaler],
    ):
        self.args = args
        self.optimizer = optimizer
        self.scaler = scaler

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, total_norm: float) -> None:
        if not self.args.no_amp:
            self.do_scaler_step()
            self.scaler.update()
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
