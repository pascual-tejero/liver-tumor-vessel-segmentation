"""

Exponential Logarithmic Dice loss
"""

import torch
import torch.nn as nn


class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # Definition of dice factor
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # The return is the dice distance
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)