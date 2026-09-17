from enum import Enum


class Kernel(str, Enum):
    proj_lse = "proj_lse"
    logsumexp = "logsumexp"
    proj_and_loss_bwd = "proj_and_loss_bwd"
    transducer_loss_bwd = "transducer_loss_bwd"

    def __str__(self) -> str:
        return self.value

    def is_forward(self) -> bool:
        return self in (Kernel.proj_lse, Kernel.logsumexp)

    def is_fused(self) -> bool:
        return self in (Kernel.proj_lse, Kernel.proj_and_loss_bwd)


class Mode(str, Enum):
    fused = "fused"
    unfused = "unfused"

    def __str__(self) -> str:
        return self.value

    def pair(self) -> tuple[Kernel, Kernel]:
        if self is Mode.fused:
            return Kernel.proj_lse, Kernel.proj_and_loss_bwd
        return Kernel.logsumexp, Kernel.transducer_loss_bwd
