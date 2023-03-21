# Multiclass segmentation

"""

Test scripts (for multi-class segmentation)
"""

import os
import copy
import collections
from time import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
from loss.DiceScore import dice_score

from net.models import ResUNet, unet, segnet, kiunet_min, kiunet_org_1, kiunet_org_2
from utilities.calculate_metrics import Metric

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# In order to calculate the two variables defined by dice_global
dice_global_0 = []
dice_global_1 = []
dice_global_2 = []
dict_missing_labels = {'0':0, '1':0, '2':0}

liver_score = collections.OrderedDict()
liver_score['dice background'] = []
liver_score['dice liver'] = []
liver_score['dice tumor'] = []

file_name = []  # File name
time_pre_case = []  # Single instance data consumption time

net = torch.nn.DataParallel(ResUNet(training=False)).cuda() # Change neural network
net.load_state_dict(torch.load(para.module_path))
net.eval()


for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    start = time()

    file_name.append(file)

    # Read CT into memory
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    
    # Truncate grayscale values that are outside the threshold
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max Normalised
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200


    # Use padding for data with too little slice
    too_small = False
    if ct_array.shape[0] < para.size:
        temp = np.zeros((para.size-ct_array.shape[0],256,256))
        ct_array = np.vstack([ct_array, temp])
        too_small = True

    # Sliding Window Sampling Prediction
    start_slice = 0
    end_slice = start_slice + para.size

    # probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)
    probability_map = np.zeros((ct_array.shape[0], 256, 256), dtype=np.float32)
    # probability_map = np.zeros((ct_array.shape[0], 128, 128), dtype=np.float32)
    out = False

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice,:,:]).cuda()

            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            # print(ct_tensor.size())
            outputs = net(ct_tensor)
            # outputs = torch.round(outputs)
            outputs = torch.argmax(outputs, dim=1)
            # print(outputs.size())

            probability_map[start_slice: end_slice,:,:] = np.squeeze(outputs.cpu().detach().numpy())

            # Due to the lack of memory, the ndarray data is kept directly here and the calculated graph is destroyed directly after saving
            del outputs      
            
            start_slice += para.stride
            end_slice = start_slice + para.size
            #print(start_slice, end_slice)
    
        if end_slice != ct_array.shape[0] or too_small == True:
            end_slice = ct_array.shape[0]
            start_slice = end_slice - para.size


            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()

            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            outputs = torch.argmax(outputs, dim=1)
            # outputs = torch.round(outputs)

            probability_map[start_slice: end_slice,:,:] = np.squeeze(outputs.cpu().detach().numpy())

            del outputs
        
        pred_seg = probability_map


    # Read the gold standard into memory
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 2] = 1 # Apply in general
    # seg_array[seg_array > 0] = 1 # Only liver detection
    # seg_array[seg_array == 1] = 0 # Only tumor detection
    # seg_array[seg_array >= 2] = 1 # Only tumor detection.

    if too_small:
      temp = np.zeros((para.size-seg_array.shape[0],256,256))
      seg_array = np.vstack([seg_array, temp])


    # Maximum connectivity extraction of the liver, removal of fine areas, and internal cavity filling
    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)

    liver_seg = liver_seg.astype(np.uint8)
    seg_array = seg_array.astype(np.uint8)

    print(f"Prediction (unique classes): {np.unique(liver_seg)} // Prediction (shape) {np.shape(liver_seg)}")
    print(f"Target (unique classes): {np.unique(seg_array)} // Target (shape) {np.shape(seg_array)}")

    dice = dice_score(seg_array, liver_seg, False)
    print(dice)

    for key in dice.keys():
      if isinstance(dice[key], str):
        dict_missing_labels[key] += 1
        continue
      elif key == '0':
        dice_global_0.append(dice[key])
      elif key == '1':
        dice_global_1.append(dice[key])
      elif key == '2':
        dice_global_2.append(dice[key])

    # Calculation of segmentation evaluation indicators

    liver_score['dice background'].append(dice['0'])
    liver_score['dice liver'].append(dice['1'])
    liver_score['dice tumor'].append(dice['2'])

    # Save the predictions as nii data
    pred_seg = sitk.GetImageFromArray(liver_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(seg.GetSpacing()) # pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)

    print(file, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


# Write evaluation indicators to the excel
liver_data = pd.DataFrame(liver_score, index=file_name)
liver_data['time'] = time_pre_case

statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
statistics.loc['mean'] = liver_data.mean()
statistics.loc['std'] = liver_data.std()
statistics.loc['min'] = liver_data.min()
statistics.loc['max'] = liver_data.max()

dice_statistics = pd.DataFrame(index=['dice score global', 'dice score background', 
  'dice score liver', 'dice score tumor'], columns=['Value'])
dice_statistics.loc['dice score background'] = np.nanmean(dice_global_0)
dice_statistics.loc['dice score liver'] = np.nanmean(dice_global_1)
dice_statistics.loc['dice score tumor'] = np.nanmean(dice_global_2)

writer = pd.ExcelWriter(para.pred_path + '/result.xlsx')
liver_data.to_excel(writer, 'Evaluation metrics')
statistics.to_excel(writer, 'Statistics')
dice_statistics.to_excel(writer, 'Dice Global Statistics')
writer.save()

# Dice global
print(f"Number of test patients: {len(os.listdir(para.test_ct_path))}")
print(f"Missing labels -> {dict_missing_labels}")
print(f"Dice global score (label 0) -> {np.nanmean(dice_global_0)}")
print(f"Dice global score (label 1) -> {np.nanmean(dice_global_1)}")
print(f"Dice global score (label 2) -> {np.nanmean(dice_global_2)}")

