"""

Test scripts (for binary segmentation)
"""

import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from net.models import kiunet_org_1, kiunet_org_2
from utilities.calculate_metrics import Metric

import parameter as para

from loss.DiceScore import dice_score

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# In order to calculate the two variables defined by dice_global
dice_intersection = 0.0  
dice_union = 0.0


file_name = []  # File name
time_pre_case = []  # Single instance data consumption time

dice_global_0 = []
dice_global_1 = []
dice_global_2 = []
dict_missing_labels = {'0':0, '1':0, '2':0}

# Defining evaluation indicators
liver_score = collections.OrderedDict()
liver_score['dice global'] = []
liver_score['dice background'] = []
liver_score['dice liver'] = []
liver_score['dice tumor'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

# Defining the network and loading parameters
net = torch.nn.DataParallel(kiunet_org_1(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    start = time() # Clock start

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

    # The CT is interpolated using the two-three times algorithm, and the interpolated array is still int16
    # ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3)

    #  Use padding for data with too little slice (put 1's until para.size is reached)
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones(para.size, 512, 512) * para.lower # temp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale))) * para.lower
        temp[0: depth] = ct_array
        ct_array = temp 
        too_small = True

    # Sliding Window Sampling Prediction
    start_slice = 0
    end_slice = start_slice + para.size - 1

    # Matrix to count the how many times a value has been predicted by the neural network
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16) 

    # Matrix to add the probability output of the neural network
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)

    # Get an outputs every 
    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1,:,:]).cuda() # 3D matrix as input
            # unsqueeze -> Returns a new tensor with a dimension of size one inserted at 
            # the specified position
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0) # Increase to 5D matrix input

            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1,:,:] += 1  
            probability_map[start_slice: end_slice + 1,:,:] += np.squeeze(outputs.cpu().detach().numpy())

            # Due to the lack of memory, the ndarray data is kept directly here and 
            # the calculated graph is destroyed directly after saving
            del outputs      
            
            # Sliding Window Sampling Prediction    
            start_slice += para.stride 
            end_slice = start_slice + para.size - 1
    
        if end_slice != ct_array.shape[0] - 1: # To make the predictions of the last slices

            # Sliding Window Sampling Prediction    
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1,:,:]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
              
            count[start_slice: end_slice + 1,:,:] += 1
            probability_map[start_slice: end_slice + 1,:,:] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs
        
        # If the probability output is higher than the threshold*(number of counts), 
        # we predict it as 1 (liver or tumor)
        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold * count)] = 1 

        # In this case, we make the prediction for the size of the image, and then we add 0's
        if too_small: 
            temp = np.zeros((depth, 512, 512), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp

    # Read the gold standard into memory
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Maximum connectivity extraction of the liver, removal of fine areas, and internal cavity filling
    pred_seg = pred_seg.astype(np.uint8) # Convert into 8-bit unsigned integer data type
    liver_seg = copy.deepcopy(pred_seg) # Creates a deep copy (independent new copy of the original array)

    # This line applies a connected component labeling algorithm to the numpy array liver_seg. 
    # Connected component labeling is a technique for identifying and labeling the connected 
    #regions in a binary image. The second argument, 4, specifies the connectivity of 
    #the labeling algorithm. In this case, a pixel is considered connected to its 4 nearest 
    #neighbors in the image plane.
    liver_seg = measure.label(liver_seg, 4)

    # This line calculates a set of properties for each labeled region in the numpy array liver_seg. 
    # The measure.regionprops function is part of the scikit-image library and can calculate a 
    # wide range of properties, including the area, perimeter, centroid, and eccentricity of
    # each labeled region. The resulting props object is a list of dictionaries, 
    # where each dictionary contains the properties for a single labeled region in liver_seg.
    props = measure.regionprops(liver_seg)
    
    # Find the largest labeled region
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    # Assign labels
    temp = np.zeros_like(liver_seg)
    temp[np.where(liver_seg != max_index)] = 1  # This area will be liver and tumor
    temp[np.where(liver_seg == max_index)] = 0  # The largest area will be background
    liver_seg = temp

    liver_seg = liver_seg.astype(np.bool) # Convert into boolean data type

    # This line applies a morphological operation to liver_seg that removes small 
    #holes in the labeled regions. The morphology.remove_small_holes function is 
    #part of the scikit-image library and requires a minimum hole size (para.maximum_hole) 
    # and connectivity (2 in this case) to be specified. The in_place=True argument modifies 
    # liver_seg directly instead of returning a new array.
    morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)

    liver_seg = liver_seg.astype(np.uint8) # Convert into 8-bit unsigned integer data type

    # Print 
    print(f"Prediction (unique classes): {np.unique(liver_seg)} // Prediction (shape): {np.shape(liver_seg)}")
    print(f"Target (unique classes): {np.unique(seg_array)} // Target (shape): {np.shape(seg_array)}")

    # Calculate dice score for each label
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
    liver_metric = Metric(seg_array, liver_seg, ct.GetSpacing())

    liver_score['dice global'].append(liver_metric.get_dice_coefficient()[0])
    liver_score['dice background'].append(dice['0'])
    liver_score['dice liver'].append(dice['1'])
    liver_score['dice tumor'].append(dice['2'])
    liver_score['jacard'].append(liver_metric.get_jaccard_index())
    liver_score['voe'].append(liver_metric.get_VOE())
    liver_score['fnr'].append(liver_metric.get_FNR())
    liver_score['fpr'].append(liver_metric.get_FPR())
    liver_score['assd'].append(liver_metric.get_ASSD())
    liver_score['rmsd'].append(liver_metric.get_RMSD())
    liver_score['msd'].append(liver_metric.get_MSD())

    dice_intersection += liver_metric.get_dice_coefficient()[1]
    dice_union += liver_metric.get_dice_coefficient()[2]

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
dice_statistics.loc['dice score global'] = dice_intersection / dice_union
dice_statistics.loc['dice score background'] = np.nanmean(dice_global_0)
dice_statistics.loc['dice score liver'] = np.nanmean(dice_global_1)
dice_statistics.loc['dice score tumor'] = np.nanmean(dice_global_2)

writer = pd.ExcelWriter(para.pred_path + '/result.xlsx')
liver_data.to_excel(writer, 'Evaluation metrics')
statistics.to_excel(writer, 'Statistics')
dice_statistics.to_excel(writer, 'Dice Global Statistics')
writer.save()

# Print dice scores
print(f"Number of test patients: {len(os.listdir(para.test_ct_path))}")
print(f"Missing labels -> {dict_missing_labels}")

print('Dice score global:', dice_intersection / dice_union)
print(f"Dice score global (label 0) -> {np.nanmean(dice_global_0)}")
print(f"Dice score global (label 1) -> {np.nanmean(dice_global_1)}")
print(f"Dice score global (label 2) -> {np.nanmean(dice_global_2)}")

