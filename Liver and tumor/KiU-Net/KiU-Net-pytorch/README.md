# Liver segmengtation using KiU-Net:

This code is built on top of [MICCAI-Lits17](https://github.com/assassint2017/MICCAI-LITS2017)

## Requirements:

```bash
numpy==1.14.2
torch==1.0.1.post2
visdom==0.1.8.8
pandas==0.23.3
scipy==1.0.0
tqdm==4.40.2
scikit-image==0.13.1
SimpleITK==1.0.1
pydensecrf==1.0rc3
```

## Dataset

Liver tumor Segmentation Challenge (LiTS) contain 131 contrast-enhanced CT images provided by hospital around the world. 3DIRCADb dataset is a subset of LiTS dataset with case number from 27 to 48. we train our model with 111 cases from LiTS after removeing the data from 3DIRCADb and evaluate on 3DIRCADb dataset. For more details about the dataset: https://competitions.codalab.org/competitions/17094

## Changes from the original code

Our architecture supports a wide variety of loss functions, including Cross Entropy Loss, Dice Loss, and Focal Loss, which can be selected by the user. However, depending on the chosen loss function, modifications may be required to the architecture output and dataset class. For this repository, our architecture outputs a 3-dimensional tensor, with each dimension representing one of the three classes, and is evaluated using a weighted Cross Entropy Loss. The loss function can be visualized using the tensorboard library, which needs to be installed beforehand.

In addition, we have included data augmentation capabilities, which are customizable by the user. The original dataset is first loaded into the data loader, and then a set of data augmentations (such as horizontal flip, random rotation, and affine transformation) are applied and concatenated to the dataloader. This allows for greater flexibility and variety in the training data.

In the validation file ()...



## Running the code

Check parameter.py to change the data directory and other settings. The model files can be found in in "net/models.py". Use train.py to train the network. Use val.py to get the predictions and performance metrics.
  
