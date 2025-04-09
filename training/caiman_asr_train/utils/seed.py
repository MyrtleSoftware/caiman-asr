import random

import numpy as np
import sentencepiece
import torch
from numpy.random import Generator


def set_seed(seed, local_rank=0) -> Generator:
    """Set random seed and return a random generator with fixed seed."""

    torch.manual_seed(seed + local_rank)

    np.random.seed(seed + local_rank)

    sentencepiece.SetRandomGeneratorSeed(seed + local_rank)

    random.seed(seed + local_rank)

    return np.random.default_rng(seed=seed)
