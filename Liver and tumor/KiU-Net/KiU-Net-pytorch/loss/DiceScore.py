import numpy as np
import torch

def dice_score(y_true, y_pred, train):

    # Training part and multiclass segmentation
    if train == True:
        if (type(y_true).__module__ != np.__name__ and (len(y_true.size()) > 4)) or (y_pred.size()[1] == 3): 
            if((y_true.size()[-1] == 3)): # Target with one hot encoding
                y_true = torch.permute(y_true, (0, 4, 1, 2, 3))
                y_true = torch.argmax(y_true, dim=1) # One-hot encoding
            y_true = y_true.squeeze().cpu().detach().numpy()
            y_pred = torch.argmax(y_pred, dim=1) 
            y_pred = y_pred.squeeze().cpu().detach().numpy()

            # print(np.count_nonzero(y_pred == 1), np.count_nonzero(y_true == 1))
            # print(np.count_nonzero(y_pred == 2), np.count_nonzero(y_true == 2))
            # print(np.shape(y_pred), np.shape(y_true))
            # print(np.unique(y_pred), np.unique(y_true))

        # Training Only binary segmentation
        elif type(y_true).__module__ != np.__name__: 
            y_true = y_true.squeeze().cpu().detach().numpy()
            y_pred = torch.round(y_pred)
            y_pred = y_pred.squeeze().cpu().detach().numpy()
            # print(np.count_nonzero(y_pred == 1), np.count_nonzero(y_true == 1))
            # print(np.count_nonzero(y_pred == 2), np.count_nonzero(y_true == 2))
            # print(np.shape(y_pred), np.shape(y_true))
            # print(np.unique(y_pred), np.unique(y_true))    
    
    # print(np.shape(y_true), type(y_true))
    # print(np.unique(y_true), np.unique(y_pred))
        
    # Find the unique labels in each matrix
    labels = [0,1,2]
    dice = {}

    # Iterate over the labels
    for label in labels:

        # Find the indices of the label in each matrix
        y_true_indices = y_true == label
        y_pred_indices = y_pred == label

        # If at least one element is True
        if y_true_indices.any(): # The any() function returns True if any item in an iterable are true, otherwise it returns False.

            # Calculate the number of true positives for the label
            true_positives = np.sum(np.logical_and(y_true_indices, y_pred_indices))

            # Update the numerator and denominator
            numerator = (2 * true_positives) 
            denominator = (np.sum(y_true_indices) + np.sum(y_pred_indices)) 

            # Calculate the Dice
            dice[str(label)] = round(numerator/denominator,5)
            
        else:
            dice[str(label)] = np.nan


    return dice

