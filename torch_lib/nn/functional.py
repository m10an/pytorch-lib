import torch
import torch.nn.functional as F
from . import _functions

mish = _functions.Mish.apply
cov_pool = _functions.CovPool.apply
newton_schulz_sqrt = _functions.NewtonSchulzSqrt.apply
triuvec = _functions.Triuvec.apply
shake_drop = _functions.ShakeDrop.apply


def gem(x, p=3, eps=1e-6):
    return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1)))
