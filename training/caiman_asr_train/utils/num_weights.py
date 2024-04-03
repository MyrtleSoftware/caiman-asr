import torch.nn as nn


def num_weights(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
