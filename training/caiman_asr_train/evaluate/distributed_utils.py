#!/usr/bin/env python3

import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Union


@beartype
def sum_across_gpus(number: Union[int, float]) -> Union[int, float]:
    number_tensor = torch.tensor(number).cuda()
    dist.all_reduce(number_tensor)
    return number_tensor.item()
