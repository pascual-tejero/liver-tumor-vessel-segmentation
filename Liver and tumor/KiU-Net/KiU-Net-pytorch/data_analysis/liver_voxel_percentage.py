"""

View the number of pixels in the liver region as a percentage of the slice containing only the liver region
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from tqdm import tqdm
import SimpleITK as sitk

import parameter as para

total_point = 0.0
total_liver_point = 0.0

for seg_file in tqdm(os.listdir(para.train_seg_path)):

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, seg_file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice or 2 in slice:
            liver_slice += 1

    liver_point = (seg_array > 0).astype(int).sum()

    print('precent:{:.4f}'.format(liver_point / (liver_slice * 512 * 512) * 100))

    total_point += (liver_slice * 512 * 512)
    total_liver_point += liver_point

print(total_liver_point / total_point)

# Training set 6.99%
# Test set 6.97%
