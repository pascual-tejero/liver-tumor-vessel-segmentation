# Liver and tumor segmengtation using KiU-Net:

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

Liver tumor Segmentation Challenge (LiTS) contain 131 contrast-enhanced CT images provided by hospital around the world. 3DIRCADb dataset is a subset of LiTS dataset with case number from 27 to 48. we train our model with 111 cases from LiTS after removeing the data from 3DIRCADb and evaluate on 3DIRCADb dataset. For more details about the dataset: https://competitions.codalab.org/competitions/17094

## Changes from the original code

In _net_ folder, you can see the modified architectures for the multi-class segmengtation task: ResUNet, UNet, SegNet, kiunet_min and kiunet_org_2). The only architecture which is equal to the original code is kiunet_org_1 which peforms a binary segmentation task.

Our architecture supports a wide variety of loss functions, including Cross Entropy Loss, Dice Loss, Focal Loss for multi-class segmentation or Tversky Loss for binary segmentation, which are in the _loss_ folder (_Loss_Multiclass.py_ file or _TverskyLoss.py_). However, depending on the chosen loss function, modifications may be required to the architecture output and dataset class. For this repository, our architecture outputs a 3-dimensional tensor, with each dimension representing one of the three classes, and is evaluated using a weighted Cross Entropy Loss. The loss function can be visualized using the tensorboard library, which needs to be installed beforehand.

In addition, we have included data augmentation capabilities in _dataset/dataset.py_ file, which are customizable by the user. The original dataset is first loaded into the data loader, and then a set of data augmentations (such as horizontal flip, random rotation, and affine transformation) are applied and concatenated to the dataloader. This allows for greater flexibility and variety in the training data.

We added two different validation files (_val_binary_seg.py_ and _val_multiclass_seg.py_) for binary or multi-class segmentation task respectively. In these files, we evaluate the performance of the neural network with the dice score parameter whose function implemented can be seen in the _loss/DiceScore.py_ file. Additionally, these files create automatically a Excel file with all the results for each patients and class label, aparto from the mean, standard deviation, minimum, maximum among other important parameters.

## Preprocessing

First, it is necessary to make a preprocessing of the dataset you are going to use to train and test the neural network. This preprocessing can be found in _data_prepare/get_training_set.py_ which makes a downsampling and resampling of the CT and mask files. We change it from the original code where no downsampling and resampling was done for the maks files.


## Running the code

Go to _parameter.py_ to change the directory where the training and test dataset is. Also, you can change the hyperparameters for the model (epochs, learning rate, learning rate decay, size...). Run the train.py to train the neural network selected in _net/models.py_ and run _val_binary_seg.py_ and _val_multiclass_seg.py_ files to test the neural network with your test dataset. The models are designed for an image size of 256x256, you may need to change the architecture with a different image size.



  
