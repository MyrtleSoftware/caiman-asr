from enum import Enum

import torch


class PipelineType(Enum):
    TRAIN = 1
    VAL = 2


TRAIN = PipelineType.TRAIN
VAL = PipelineType.VAL

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
