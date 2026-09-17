import torch
from beartype import beartype
from torch.torch_version import TorchVersion

MIN_TORCH_FOR_TILELANG = "2.7"


@beartype
def sm80_supported() -> bool:
    """
    Our Tilelang kernels require a CUDA GPU of compute capability >= 8.0 (Ampere and
    newer). On older GPUs (e.g. Turing sm_75) the custom CUDA LSTM and the Triton
    transducer loss are used instead.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


@beartype
def torch_supports_tilelang() -> bool:
    return TorchVersion(torch.__version__) >= MIN_TORCH_FOR_TILELANG


@beartype
def tilelang_supported() -> bool:
    return sm80_supported() and torch_supports_tilelang()
