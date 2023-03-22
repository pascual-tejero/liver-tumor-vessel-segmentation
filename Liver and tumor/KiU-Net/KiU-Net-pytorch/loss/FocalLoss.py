import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        device = inputs.device

        num_classes = inputs.shape[1]
        class_weights = self.compute_class_weights(targets, num_classes)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='none')

        loss = loss_fn(inputs, targets.long())

        if self.alpha is not None:
            probs = F.softmax(inputs, dim=1)
            pt = probs * targets + (1 - probs) * (1 - targets)
            w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            w = w * (1 - pt).pow(self.gamma)
            loss = loss * w

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

    def compute_class_weights(self, targets, num_classes):
        num_samples = targets.shape[0]
        class_weights = torch.zeros(num_classes)

        for i in range(num_classes):
            num_pixels = torch.sum(targets == i)
            if num_pixels > 0:
                class_weights[i] = num_samples / num_pixels
        return class_weights

           

