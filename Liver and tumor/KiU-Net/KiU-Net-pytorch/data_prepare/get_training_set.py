"""
Obtain a training dataset that can be used to train the network
Takes about 40 minutes and generates training data of about 3G in size
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

import parameter as para


if os.path.exists(para.training_set_path):
    shutil.rmtree(para.training_set_path) # Removes everything in training_set_path

new_ct_path = os.path.join(para.training_set_path, 'ct')
new_seg_dir = os.path.join(para.training_set_path, 'seg')

os.mkdir(para.training_set_path)
os.mkdir(new_ct_path)
os.mkdir(new_seg_dir)

start = time()
for file in tqdm(os.listdir(para.train_ct_path)):

    # Read CT and gold standards into memory
    print(os.path.join(para.train_ct_path, file))
    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Fusion of the gold standard labels for liver and liver tumours into one
    # seg_array[seg_array > 0] = 1

    # Truncate grayscale values that are outside the threshold
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # Downsampling and resampling of the CT data in cross-section, adjusting the spacing of the z-axis to 1mm for all data
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)
    # seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, 1, 1), order=0)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)

    # Find the slice at the beginning and end of the liver region and expand the slice outwards in each direction
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # Slice expansion in both directions
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

    # If the number of slice left at this point is less than the size, the data will be discarded directly, so there is very little data left, so don't worry about it.
    if end_slice - start_slice + 1 < para.size:
        print('!!!!!!!!!!!!!!!!')
        print(file, 'have too little slice', ct_array.shape[0])
        print('!!!!!!!!!!!!!!!!')
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    # Final saving of data as nii
    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)
    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    # new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], para.slice_thickness))
    new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
    # sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file))