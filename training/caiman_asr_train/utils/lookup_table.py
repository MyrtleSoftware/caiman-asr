import warnings

import torch
from beartype import beartype


@beartype
def build_lookup_table(hashmap: dict[int, int]) -> torch.Tensor:
    """Builds a vectorized lookup table from a dictionary.
    It validates all keys are small positive integers."""
    max_id = max(hashmap.keys())
    if max_id >= 100:
        warnings.warn(
            f"Unexpectedly large max_id ({max_id}). Prefix token_ids should be "
            "small integers (expected < 100). This may cause high memory usage.",
            UserWarning,
            stacklevel=2,
        )
    invalid_sentinel = torch.iinfo(torch.long).max
    lookup_tensor = torch.full((max_id + 1,), invalid_sentinel, dtype=torch.long)
    for k, v in hashmap.items():
        lookup_tensor[k] = v
    return lookup_tensor
