"""

Jaccard loss
"""

import torch
import torch.nn as nn


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # Definition of jaccard factor
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # The return is the jaccard distance
        return torch.clamp((1 - dice).mean(), 0, 1)
