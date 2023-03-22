# Liver and tumor segmentation:

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

Additionally, you need to install the following libraries:  
```bash
pip install SimpleITK
pip install torchmetrics
pip install tensorboard
```

## Dataset

Mainly, we used Liver tumor Segmentation Challenge (LiTS) dataset containing 131 contrast-enhanced CT images provided by hospital around the world. We discarded those 14 patients with no or little tumor segmentation or with medical artifacts (30, 32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119), and divided the dataset into train and test sets.
- Patients in training set (101): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 and 130.
- Patients in test set (16): 27, 28, 29, 31, 33, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46 and 48.

Note that for the cases of cross-validation or ablation study (in _Results_ folder) we use a different configuration of train and test sets than the this one.

## Changes from the original code

In _net_ folder, you can see the modified architectures for the multi-class segmengtation task: ResUNet, UNet, SegNet, kiunet_min and kiunet_org_2). The only architecture which is equal to the original code is kiunet_org_1 which peforms a binary segmentation task.

Our architecture supports a wide variety of loss functions, including Cross Entropy Loss, Dice Loss, Focal Loss for multi-class segmentation or Tversky Loss for binary segmentation, which are in the _loss_ folder (_Loss_Multiclass.py_ file or _TverskyLoss.py_). However, depending on the chosen loss function, modifications may be required to the architecture output and dataset class. For this repository, our architecture outputs a 3-dimensional tensor, with each dimension representing one of the three classes, and is evaluated using a weighted Cross Entropy Loss. The loss function can be visualized using the tensorboard library, which needs to be installed beforehand.

In addition, we have included data augmentation capabilities in _dataset/dataset.py_ file, which are customizable by the user. The original dataset is first loaded into the data loader, and then a set of data augmentations (such as horizontal flip, random rotation, and affine transformation) are applied and concatenated to the dataloader. This allows for greater flexibility and variety in the training data.

We added two different validation files (_val_binary_seg.py_ and _val_multiclass_seg.py_) for binary or multi-class segmentation task respectively. In these files, we evaluate the performance of the neural network with the dice score parameter whose function implemented can be seen in the _loss/DiceScore.py_ file. Additionally, these files create automatically a Excel file with all the results for each patients and class label, aparto from the mean, standard deviation, minimum, maximum among other important parameters.

## Preprocessing

First, it is necessary to make a preprocessing of the dataset you are going to use to train and test the neural network. This preprocessing can be found in _data_prepare/get_training_set.py_ which makes a downsampling and resampling of the CT and mask files. We change it from the original code where no downsampling and resampling was done for the maks files.


## Running the code

Go to _parameter.py_ to change the directory where the training and test dataset is. Also, you can change the hyperparameters for the model (epochs, learning rate, learning rate decay, size...). Run the train.py to train the neural network selected in _net/models.py_ and run _val_binary_seg.py_ and _val_multiclass_seg.py_ files to test the neural network with your test dataset. The models are designed for an image size of 256x256, you may need to change the architecture with a different image size.



  
