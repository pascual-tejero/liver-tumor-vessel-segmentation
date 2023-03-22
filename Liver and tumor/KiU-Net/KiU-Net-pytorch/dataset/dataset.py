"""

Dataset definition script in torch
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset
import torchvision.transforms as transforms

import parameter as para
import torch.nn.functional as F


def add_gaussian_noise(img, mean=0, std=0.01, p=0.05):
    if random.uniform(0, 1) < p:
        noise = torch.randn(img.size(), dtype=torch.float32) * std + mean
        img = img + noise
    return img

  
class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, apply_transforms):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list)) # For LiTS dataset
        # self.seg_list = os.listdir(seg_dir) # For Medical Segmentation Decathlon dataset

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        self.apply_transforms = apply_transforms

        # Data transformations applied (if necessary)
        self.data_augmentation_1 = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.3, 0.3))
        ])

        self.data_augmentation_2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.3, 0.3))
        ])

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # Reading CT and gold standards into memory
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array[seg_array > 2] = 1 # Apply in general
        # seg_array[seg_array > 0] = 1 # Only liver with tumor detection (binary segmentation)
        # seg_array[seg_array == 1] = 0 # Only tumor detection (binary segmentation)
        # seg_array[seg_array == 2] = 1 # Only tumor detection (binary segmentation)

        # min max Normalised
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 48 randomly selected slice in the slice plane
        start_slice = random.randint(0, ct_array.shape[0] - para.size)
        end_slice = start_slice + para.size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]


        # After data processing, the array is converted to a tensor
        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array)
        # seg_array = F.one_hot(seg_array, num_classes=3)

        # Data augmentation
        if self.apply_transforms == 1:
            images = torch.cat((ct_array, seg_array), dim=0)
            images = self.data_augmentation_1(images)
            ct_array, seg_array = torch.split(images, [ct_array.shape[0], seg_array.shape[0]], dim=0)
        elif self.apply_transforms == 2:
            images = torch.cat((ct_array, seg_array), dim=0)
            images = self.data_augmentation_2(images)
            ct_array, seg_array = torch.split(images, [ct_array.shape[0], seg_array.shape[0]], dim=0)
        
        ct_array = ct_array.unsqueeze(0)

        # Apply data augmentation to the data
        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)
