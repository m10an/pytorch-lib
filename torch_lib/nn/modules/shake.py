import torch
import torch.nn as nn
from ..functional import shake_drop


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return shake_drop(x, self.training, self.p_drop, self.alpha_range)
