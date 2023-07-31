# Copyright (c) 2022 Myrtle.ai

import torch


# The 8 in Hardsigmoid is easier/cheaper in hardware than non-powers-of-2
def Hardsigmoid(x):
    return torch.clamp(0.5 + x / 8.0, min=0.0, max=1.0)


# Using Hardtanh in a @torch.jit.script was faster than using torch.nn.functional.hardtanh
def Hardtanh(x):
    return torch.clamp(x, min=-1.0, max=1.0)
