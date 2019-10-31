from torch import nn
from ..functional import mish


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)
