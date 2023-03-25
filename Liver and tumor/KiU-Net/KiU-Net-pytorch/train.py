import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss
from loss.DiceScore import dice_score
from loss.Loss_Multiclass import Loss_Multiclass
from loss.FocalLoss import FocalLoss

from net.models import net_model

import parameter as para

from torch.utils.tensorboard import SummaryWriter


# Set CUDA and cuDNN parameters for GPU computation.
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# Create the model for segmentation task
net = torch.nn.DataParallel(net_model).cuda()
net.train()

# Create a data loader with no data augmentation
# train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))
# train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# Create a data loader with data augmentation
train_ds_original = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'), apply_transforms=0)
train_ds_transformed_2 = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'), apply_transforms=2)
train_ds_concat = torch.utils.data.ConcatDataset([train_ds_original, train_ds_transformed_2])

train_dl = DataLoader(train_ds_concat, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)


# Define a list of loss functions
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
# loss_func = TverskyLoss() # Binary segmentation loss
loss_func = Loss_Multiclass() # Multi-class segmentation loss
# loss_func = FocalLoss() # Multi-class segmentation loss

# Define the optimization algorithm.
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# Create a scheduler for learning rate decay during training
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# Set the alpha parameter for loss function weighting
alpha = para.alpha

# Tensorboard
writer = SummaryWriter('logs')

# Timer
start = time()

# Loop over the number of epochs specified in the para.Epoch parameter
for epoch in range(para.Epoch): 

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        # Load input and target images
        ct = ct.cuda()
        seg = seg.cuda()

        # Model prediction
        outputs = net(ct)

        # Compute loss at every layer of the model
        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        # Weighted loss
        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())
        writer.add_scalar('Loss/train', loss4.item(), epoch) #Tensorboard

        # Update the model parameters using the optimizer
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
        #           .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))  
        # print(dice_score(seg,outputs[3], True))
        
        # Print training progress 
        if step % 5 == 0:           
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
            print(dice_score(seg, outputs[3], True))
        # if step % 10 == 0:
            

    mean_loss = sum(mean_loss) / len(mean_loss)

    # Save the model parameters after every 50 epochs.
    if epoch % 50 == 0 :
        torch.save(net.state_dict(), 'KiU-Net-pytorch/LiTS/saved-networks/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    # Reduce the weight of the auxiliary loss after every 40 epochs.
    if epoch % 40 == 0 and epoch != 0:
        alpha *= 0.8

