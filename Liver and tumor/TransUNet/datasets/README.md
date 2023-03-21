# Preprocessing Data

With this notebook you can process the Train and Test data to be able to run in TransUNet network.

1. Access to the LiTS organ dataset:
   1. Sign up in the [official LiTS website](https://competitions.codalab.org/competitions/17094#participate-get-data) and download the dataset. Convert them to numpy format, change the windowing image to [-200, 200], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
2. The directory structure of the whole project is as follows:

```bash
.
├── TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──LITS
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
```
