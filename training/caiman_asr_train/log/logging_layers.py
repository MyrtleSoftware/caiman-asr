import math

from beartype import beartype
from beartype.typing import List, Optional, Tuple
from torch.cuda.amp import GradScaler

from caiman_asr_train.train_utils.distributed import print_once, unwrap_ddp


@beartype
def get_logging_entries(
    model, scaler: Optional[GradScaler]
) -> Tuple[float, List[Tuple[str, float]], float]:
    """
    Returns:
    - total model gradient norm
    - per-layer info
    - log base 2 of the GradScaler's scale
    """
    # Record before the optimizer step because it may get changed
    log2_scaler = 0.0 if scaler is None else math.log2(scaler.get_scale())
    total_norm = 0.0
    tb_per_layer_logs = []
    try:
        for n, p in unwrap_ddp(model).named_parameters():
            # Only log/compute the gradnorm of parameters that aren't frozen:
            if not p.requires_grad:
                continue
            # in case of pytorch AMP compute the unscaled norm:
            if scaler is not None:
                param_grad = p.grad.data / scaler.get_scale()
            else:
                param_grad = p.grad.data

            # log weight norm and std
            weight = p.data
            weight_norm = weight.norm(2).item()
            weight_std = weight.std().item()

            tb_per_layer_logs.append((f"per-layer-weight-norm/{n}", weight_norm))
            tb_per_layer_logs.append((f"per-layer-weight-std/{n}", weight_std))

            # log grad norm and grad std
            param_norm = param_grad.norm(2).item()
            norm_max = param_grad.abs().max().item()
            param_std = param_grad.std().item()

            tb_per_layer_logs.append((f"per-layer-grad-norm/{n}", param_norm))
            tb_per_layer_logs.append((f"per-layer-grad-max/{n}", norm_max))
            tb_per_layer_logs.append((f"per-layer-grad-std/{n}", param_std))

            total_norm += param_norm**2

        total_norm = total_norm ** (1.0 / 2)
    except AttributeError as e:
        print_once(f"Exception occurred: {e}")
        total_norm = 0.0

    return total_norm, tb_per_layer_logs, log2_scaler
