import math
from argparse import Namespace

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Optional
from torch.amp import GradScaler

from caiman_asr_train.train_utils.distributed import print_once


@beartype
class OptimizerWrapper:
    """Wrapper to control the optimizer and AMP scaling during training."""

    # EMA horizon for the typical-gradient-norm estimate (~100 steps)
    NORM_EMA_DECAY = 0.99

    def __init__(
        self,
        args: Namespace,
        optimizer,
        scaler: Optional[GradScaler],
        lower_bound: Optional[float] = None,
        norm_skip_factor: Optional[float] = None,
        skip_decay_momentum: float = 0.0,
    ):
        self.args = args
        self.optimizer = optimizer
        self.scaler = scaler
        self.lower_bound = lower_bound
        self.norm_skip_factor = norm_skip_factor
        self.skip_decay_momentum = skip_decay_momentum
        # Running EMA of accepted finite gradient norms. Relative gating is
        # required because healthy norms move by orders of magnitude over a
        # run (O(8000) in early warmup vs O(10) once converged), so no static
        # threshold separates "normal" from "garbage" at every stage.
        self._norm_ema: Optional[float] = None
        # Consecutive-skip "velocity" for the momentum-augmented decay-on-skip:
        # accumulates one unit per skipped step and resets to 0 on a real step.
        self._skip_velocity: float = 0.0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, total_norm: float) -> bool:
        """
        Run the (scaler-mediated) optimizer step.

        Returns True when the underlying optimizer actually stepped, False when
        the update was skipped (the GradScaler found non-finite gradients,
        total_norm was non-finite in the no-AMP path, or total_norm exceeded
        norm_skip_factor times its running EMA). Callers use this to skip work
        that must only follow a real weight update, e.g. the EMA blend.
        """
        if self._skip_for_huge_norm(total_norm):
            self._decay_on_skip()
            return False

        if math.isfinite(total_norm) and total_norm > 0:
            # Track the typical norm over accepted steps only, so a burst of
            # rejected garbage norms cannot drag the estimate up towards them
            if self._norm_ema is None:
                self._norm_ema = total_norm
            else:
                decay = self.NORM_EMA_DECAY
                self._norm_ema = decay * self._norm_ema + (1 - decay) * total_norm

        if not self.args.no_amp:
            stepped = self.do_scaler_step()
            self.scaler.update()

            if self.lower_bound is not None:
                scale = self.scaler.get_scale()

                if scale < self.lower_bound:
                    print_once(
                        f"WARNING: grad scaler scale {scale} fell below the "
                        f"lower bound; pinning to {self.lower_bound}"
                    )
                    self.scaler.update(self.lower_bound)

            if not stepped:
                self._decay_on_skip()
            else:
                self._skip_velocity = 0.0  # a real step ends the skip run

            return stepped

        # when not using AMP test for inf / NaN gradients
        if np.isfinite(total_norm):
            self.optimizer.step()
            self._skip_velocity = 0.0  # a real step ends the skip run
            return True
        self._decay_on_skip()
        return False

    def _skip_for_huge_norm(self, total_norm: float) -> bool:
        """
        True when total_norm is finite but anomalously large vs its EMA.

        A finite-but-huge unscaled gradient norm passes the GradScaler's inf
        check, and LAMB's trust ratio then takes a full-sized step in its
        (garbage) direction regardless of the norm. Drop the update instead;
        the gradients did not overflow, so the scale is left untouched.
        Non-finite norms return False and fall through to the scaler, which
        skips AND backs the scale off.
        """
        if not self.norm_skip_factor or self._norm_ema is None:
            # None or 0 = check disabled
            return False

        if not math.isfinite(total_norm):
            return False

        if total_norm <= self.norm_skip_factor * self._norm_ema:
            return False

        print_once(
            f"WARNING: skipping optimizer step: gradient norm {total_norm:.4g} "
            f"exceeds {self.norm_skip_factor:.4g}x its running average "
            f"{self._norm_ema:.4g}"
        )
        return True

    @torch.no_grad()
    def _decay_on_skip(self) -> None:
        """
        Apply decoupled weight decay when the optimizer step is skipped, with a
        momentum term that accelerates the decay over a run of consecutive skips.

        A skip "velocity" accumulates one unit per skipped step
        (v <- skip_decay_momentum * v + 1) and scales the per-step decay, so a
        long skip episode -- where the fixed lr*weight_decay barely dents the
        (still-growing) weights -- is shrunk progressively harder. The velocity
        resets to 0 on the next real optimizer step (see step()).

        skip_decay_momentum=0 keeps v=1 every skip, i.e. the un-augmented fixed
        lr*weight_decay. momentum=1 ramps v linearly (1, 2, 3, ...); 0<momentum<1
        saturates it at 1/(1-momentum). The per-group decay is clamped to <=1 so
        the factor stays in [0, 1) and can never flip a weight's sign.
        """
        self._skip_velocity = self.skip_decay_momentum * self._skip_velocity + 1.0
        for group in self.optimizer.param_groups:
            decay = group["lr"] * group.get("weight_decay", 0.0) * self._skip_velocity
            if decay <= 0.0:
                continue
            factor = 1.0 - min(decay, 1.0)
            for p in group["params"]:
                p.mul_(factor)

    def do_scaler_step(self) -> bool:
        # pyTorch AMP step function unscales the gradients
        # if these gradients do not contain infs or NaNs, optimizer.step() is then called
        self.scaler.step(self.optimizer)
        # The scaler records its inf check per optimizer between step() and
        # update(); read it here to learn whether the optimizer really stepped
        state = self.scaler._per_optimizer_states[id(self.optimizer)]
        found_inf = sum(v.item() for v in state["found_inf_per_device"].values())
        return found_inf == 0

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
