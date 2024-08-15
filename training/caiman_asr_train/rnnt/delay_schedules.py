from beartype import beartype


@beartype
class ConstantDelayPenalty:
    """
    A scheduler that returns a constant value of delay penalty.

    Arguments:
    :penalty: fixed delay penalty value to return for every step
    """

    def __init__(self, penalty: float) -> None:
        self.curr_penalty = penalty

    def get_value(self) -> float:
        return self.curr_penalty

    def step(self, step: int) -> None:
        pass


@beartype
class LinearDelayPenaltyScheduler:
    """
    A linear delay penalty scheduler with a ramp. The penalty value is kept
    constant during warm_up steps. Afterwards, it is increased step-wise in
    a single step and then linearly until final number of steps is achieved.
    Finally, the value is kept constant for all further steps.

    Arguments:
    :warmup_steps: number of warm_up steps before step-wise increase,
        recommend value 5000 or when model reaches about 30% WER
    :warmup_penalty: penalty value during warm_up period,
        setting to 0.0 stabilizes training and improves final WER
    :ramp_penalty: value to increase penalty step-wise at step=`warmup_steps`+1,
        recommend 0.007 to avoid steep EL increase that occurs at the start of training
    :final_steps: final number of steps to keep increasing penalty linearly
        recommend 25000
    :final_penalty: penalty value to reach at step=`final_steps`+1 and keep afterwards
        recommend 0.01-0.015 to keep EL at or below 200ms
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_penalty: float,
        ramp_penalty: float,
        final_steps: int,
        final_penalty: float,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.final_steps = final_steps
        self.warmup_penalty = warmup_penalty
        self.ramp_penalty = ramp_penalty
        self.final_penalty = final_penalty

    def get_value(self) -> float:
        return self.curr_penalty

    def step(self, step: int) -> None:
        if step < self.warmup_steps:
            self.curr_penalty = self.warmup_penalty
        elif step > self.final_steps:
            self.curr_penalty = self.final_penalty
        else:
            rate = (self.final_penalty - self.ramp_penalty) / (
                self.final_steps - self.warmup_steps
            )
            self.curr_penalty = self.ramp_penalty + (step - self.warmup_steps) * rate
