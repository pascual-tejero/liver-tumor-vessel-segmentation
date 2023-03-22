"""

See the proportion of liver region slice to overall slice
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from tqdm import tqdm
import SimpleITK as sitk

import parameter as para

total_slice = 0.0
total_liver_slice = 0.0

for file in tqdm(os.listdir(para.test_seg_path)):

    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice or 2 in slice:
            liver_slice += 1

    total_slice += seg_array.shape[0]
    total_liver_slice += liver_slice

    print('precent:{:.4f}'.format(liver_slice / seg_array.shape[0] * 100))

print(total_liver_slice / total_slice)

# % of slice with liver in the training set overall: 30.61%
# % of slice containing liver in the test set overall: 73.46%
