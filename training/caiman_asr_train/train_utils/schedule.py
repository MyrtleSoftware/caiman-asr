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
