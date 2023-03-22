"""

Dice Loss Multitask segmentation task
"""

import torch
import torch.nn as nn
from torchmetrics import Dice

class Loss_Multiclass(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """"
        Cross-entropy loss:
        """
        # print(pred.size(), target.size())

        weights = torch.tensor([0.01, 0.39, 0.6]).to(device)
        loss = nn.CrossEntropyLoss(weight=weights)
        output = loss(pred,target.long())
        return output

        
        """
        Weighted Dice loss:
        
        Calculate dice score for each label manually or globally
        """
        # Swap dimensions
        # target = torch.permute(target, (0, 4, 1, 2, 3))

        # Apply softmax to prediction
        # m = nn.Softmax(dim=1)
        # pred = m(pred)

  

        # Tversky loss
        # Definition of dice factor
        # dice_score = torch.tensor([0.0, 0.0, 0.0])
        # for i in range(0,3):
        #   pred_label = pred[:,i,:,:,:]
        #   target_label = target[:,i,:,:,:]
        #   dice_score[i] = (pred_label * target_label).sum() / ((pred_label * target_label).sum() +
        #                                 0.3 * (pred_label * (1 - target_label)).sum() + 
                                        # 0.7 * ((1 - pred_label) * target_label).sum())

        # The return is the dice distance
        # return torch.clamp((1 - dice_score).mean(), 0, 2)

        """
        Dice loss:
        
        Apply torch.argmax() to prediction, and apply the loss function Dice()
        """
        # It does not work because:
        #RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
 

        # pred = torch.argmax(pred, dim=1)
        # dice = Dice(average='micro').to(device)
        # dice_result = dice(pred, target).detach()
        # return 1 - dice_result