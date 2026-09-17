import math
from abc import ABC, abstractmethod

from beartype import beartype
from beartype.typing import Any, Dict, Optional


class Schedule(ABC):
    """
    Abstract class for scheduling parameters over a train.
    """

    @abstractmethod
    @beartype
    def step(self, train_step: int, *, hints: Optional[Dict[str, Any]] = None) -> float:
        """
        Set the value of the schedule.

        Arguments:
            train_step: the current step in the training loop.
            hints: extra information that can be used to compute the schedule.

        Returns:
            The value of the schedule at the new step.
        """
        pass

    @abstractmethod
    @beartype
    def value(self) -> float:
        """
        Get the value of the schedule at the current step.
        """
        pass


class ConstantSchedule(Schedule):
    """
    A scheduler that returns a constant value.

    Arguments:
        value: fixed value to return for every step
    """

    @beartype
    def __init__(self, value: float) -> None:
        self._value = value

    @beartype
    def value(self) -> float:
        return self._value

    @beartype
    def step(self, step: int, *, hints: Optional[Dict[str, Any]] = None) -> float:
        return self._value


class StepSchedule(Schedule):
    @beartype
    def __init__(
        self,
        initial_value: float,
        final_value: float = 1.0,
        toggle_step: Optional[int] = None,
        wer_threshold: Optional[float] = None,
    ) -> None:
        """
        A schedule that jumps from an initial value to a final value
        after a specific step or when the WER is below a threshold.

        Arguments:
            initial_value: the initial value of the schedule.
            final_value: the final value of the schedule.
            toggle_step: if present the step at which the schedule is set to final_value.
            wer_threshold: if present and the WER is below this value, the schedule
                is set to final_value.
        """
        self.initial_value = initial_value
        self.final_value = final_value

        self.toggle_step = toggle_step
        self.wer_threshold = wer_threshold

        self.set = False

        if toggle_step is None and wer_threshold is None:
            raise ValueError(
                "StepSchedule is not set to change at any step or WER threshold"
            )

    @beartype
    def step(self, train_step: int, *, hints: Optional[Dict[str, Any]] = None) -> float:
        if self.set:
            return self.value()

        if self.toggle_step is None and self.wer_threshold is not None:
            if hints is None or "wer" not in hints:
                raise ValueError(
                    "StepSchedule expecting WER in hints but it was not found."
                )

        if self.wer_threshold is not None and hints is not None and "wer" in hints:
            if hints["wer"] < self.wer_threshold:
                self.set = True

        if self.toggle_step is not None and train_step >= self.toggle_step:
            self.set = True

        return self.value()

    def value(self) -> float:
        if self.set:
            return self.final_value

        return self.initial_value


@beartype
class DPSchedule(Schedule):
    """
    Delay penalty scheduler that ramps from initial value to final value
    based on WER thresholds.
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        final_value: float = 0.01,
        wer_threshold_initial: float = 0.95,
        wer_threshold_final: float = 0.4,
        interpolation_steps: int = 1000,
    ) -> None:
        """
        Args:
            initial_value:  Initial penalty at the start of training.
            final_value: Penalty value to ramp up to and continue to the end of training.
            wer_threshold_initial: Penalty stays at initial_value for WERs above this value.
            wer_threshold_final: Penalty jumps to final_value for WERs below this value.
            interpolation_steps: Number of training steps to spend smoothly transitioning
                                 penalty values between WER intervals. Should be
                                 set to the same value as WER interval
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.wer_threshold_initial = wer_threshold_initial
        self.wer_threshold_final = wer_threshold_final
        self.interpolation_steps = interpolation_steps

        # The value actually used by the loss function. Updated every step.
        self._current_value = initial_value

        # The penalty value at the moment the last WER eval arrived.
        self._source_value = initial_value

        # The penalty value to interpolate towards.
        # Set each time a new WER arrives.
        self._target_value = initial_value

        self._interp_start_step = 0
        self._last_seen_wer = None

    @beartype
    def step(self, train_step: int, *, hints: Optional[Dict[str, Any]] = None) -> float:

        if hints is not None and "wer" in hints:
            current_wer = hints["wer"]
            if current_wer != self._last_seen_wer:
                self._last_seen_wer = current_wer
                # desired penalty value
                new_target = self._compute_target_from_wer(current_wer)
                new_target = max(new_target, self._current_value)

                self._source_value = self._current_value
                self._target_value = new_target
                self._interp_start_step = train_step

        elapsed = train_step - self._interp_start_step
        progress = min(elapsed / self.interpolation_steps, 1.0)

        # Cosine schedule
        beta = 0.5 * (1 - math.cos(math.pi * progress))
        self._current_value = self._source_value + (
            (self._target_value - self._source_value) * beta
        )
        return self._current_value

    def _compute_target_from_wer(self, wer: float) -> float:
        """
        Compute penalty value based on WER value.
        """
        if wer >= self.wer_threshold_initial:
            return self.initial_value

        elif wer <= self.wer_threshold_final:
            return self.final_value

        else:
            # cosine schedule
            progress = (wer - self.wer_threshold_initial) / (
                self.wer_threshold_final - self.wer_threshold_initial
            )
            beta = 0.5 * (1 - math.cos(math.pi * progress))
            return self.initial_value + (self.final_value - self.initial_value) * beta

    def value(self) -> float:

        return self._current_value
