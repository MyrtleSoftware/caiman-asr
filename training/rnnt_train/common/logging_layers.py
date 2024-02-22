from beartype.typing import List, Optional, Tuple
from torch.cuda.amp import GradScaler

from rnnt_train.common.helpers import print_once


def get_logging_entries(
    model, scaler: Optional[GradScaler]
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Return total model gradient norm and per-layer info.
    """
    total_norm = 0.0
    tb_per_layer_logs = []
    try:
        for n, p in getattr(model, "module", model).named_parameters():
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

    return total_norm, tb_per_layer_logs
