from torch import nn


class SEContextGating(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.BatchNorm1d(channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        input = x
        gating = self.gating(x)
        return input * gating
