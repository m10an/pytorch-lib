import torch
import torch.nn as nn
import torch.nn.init as init
from . import Identity
from .. functional import cov_pool, newton_schulz_sqrt, triuvec
    

class iSQRTCOVPool(nn.Module):
    """
    iSQRT-COV pooling layer.
    Parameters:
    ----------
    input_features : int
        Number of input features
    num_iter : int, default 5
        Number of iterations (num_iter > 1) recommended 3.
    """
    def __init__(self,
                 input_features=256,
                 num_iter=5):
        super().__init__()

        self.num_iter = num_iter
        self.out_shape = input_features * (input_features + 1) // 2

    def forward(self,x):
        x = cov_pool(x)
        x = newton_schulz_sqrt(x, self.num_iter)
        x = triuvec(x)
        return x


class DimReduction(nn.Module):
    """
    Dimension reduction layer.
    Parameters:
    ----------
    input_features : int
        Number of input features
    dimension_reduction : int, default 256
        Number of features to use
    """
    def __init__(self,
                 input_features=1024,
                 dimension_reduction=256,
                 activation='relu'):
        super().__init__()

        self.out_shape = dimension_reduction
        self.conv = nn.Conv2d(input_features, dimension_reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(dimension_reduction)
        if   activation is None:
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(True)
        else:
            raise KeyError()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ProgressiveDimReduction(nn.Module):
    """
    https://arxiv.org/pdf/1703.08050.pdf
    https://arxiv.org/pdf/1904.06836.pdf 
    Based on table 4
    Dimension reduction layer.
    Parameters:
    ----------
    input_features : int
        Number of input features
    dimension_reduction first : int,default 512
        Number of features after first dim red
    """
    def __init__(self,
                 input_features=1024,
                 dimension_reduction_first=512,
                 dimension_reduction_last=256,
                 activation='relu'):
        super().__init__()
  
        self.out_shape = dimension_reduction_last
        self.dr_block_1 = DimReduction(input_features, dimension_reduction_first, activation)
        self.dr_block_2 = DimReduction(dimension_reduction_first, dimension_reduction_last, activation)

    def forward(self,x):
        x = self.dr_block_1(x)
        x = self.dr_block_2(x)
        return x

