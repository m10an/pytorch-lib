import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = torch.tensor(weights, dtype=torch.float).cuda()
        self.reduction = reduction

    def forward(self, input, target):
        
        weights = self.weights[target.data.view(-1).long()].view_as(target)
        loss = self.criterion(input,target)
        loss = loss * weights

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss
