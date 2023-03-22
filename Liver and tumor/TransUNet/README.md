# TransUNet - Liver and Tumor Segmentation

This folder holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf) and adapted to run it in LITS dataset.

The code is able to run it locally or in polyaxon. However, is recommend it to run it in polyaxon for better performance.

Before training the model on LITS dataset, the model was trained and test on the Synapse dataset which is the original dataset that was used in TransUNet architecture. After obtaining similar results as in the paper, the code was modified to be able to run it in Polyaxon and LITS data set. This changes can be visualized in the files with a comment that says `# CHANGE`

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, the pre-processed is in NAS folders "Natalia AP/TransUnet/data/LITS".

### 3. Parameter Configuration

In the file ["./config/config_dncnn.yaml"](config) it is possible to set if the code is run in Polyaxon (`on_polyaxon: True`) or Locally `on_polyaxon: False`.

In addition, it is possible to set:
- `lits_dataset`: Folder path where is located the LITS dataset (locally or Polyaxon)
- `synapse_dataset`: Folder path where is located Synapse dataset (locally or Polyaxon)
- `pretrained_model`: Folder path where is located the pretained model
- `gpus`: GPU to use
- `num_workers`: Number of workers to train the model

In addition, in the folder ["./lists/lists_LITS"](/lists/lists_LITS) you need to add the file "train.txt" and "test_vol.txt" which contain the name of the files that are going to be use for training and testing respectively.

### 4. Environment

Please prepare an environment with python=3.7 and cuda 11 to be able to run it in polyaxon, and the requirements are located in "requirements.txt" for the dependencies.


### 5. Train/Test

- Run the polyaxon file "polyaxonfile.yaml". To run the training, comment line 32 of the file and uncomment line 31 of the file, and set or change the different hyperparameters.
    - `--vit_name`: pretained model used.
    - `--dataset`: Dataset to train
    - `--base_lr`: Learning rather
    - `--max_epochs`: max_epochs
    - `--img_size`: size of the image
    - `--batch_size`: batch size

```bash
cmd: CUDA_VISIBLE_DEVICES=0 python -u train.py --dataset LITS --vit_name R50-ViT-B_16 --base_lr 0.01 --max_epochs 15 --img_size 256 --batch_size 20
```
- Run the polyaxon file "polyaxonfile.yaml". It supports testing for both 2D images and 3D volumes. 
- To run the Testing, comment line 31 of the file and uncomment line 32 of the file, and set or change the different hyperparameters.
    - `--vit_name`: pretained model used.
    - `--dataset`: Dataset to train
    - `--base_lr`: Learning rather
    - `--max_epochs`: max_epochs
    - `--img_size`: size of the image
    - `--batch_size`: batch size
    - `--model_time`: Folder where the model is save and the time where it was created
    - `--is_savenii`: Add if you want to save the resulting images,prediction and ground truth of your testing data.

```bash
cmd: python test.py --dataset LITS --vit_name R50-ViT-B_16 --base_lr 0.01 --max_epochs 15 --img_size 256 --batch_size 20 --model_time 20230321_07_32_54 --is_savenii
```
The output and the saved files are located in NAS cluster in the folder

## Results

### Cross Validation

It was used 5 folding cross validation, with 80% of the data for training and 20% of the data for testing.

In the folder ["./lists/lists_LITS/train/cv5"](/lists/lists_LITS/Train/cv5) and ["./lists/lists_LITS/test/cv5"](/lists/lists_LITS/Test/cv5) there are the the files for each folding. To train each folding individually, the train and test file has to be copy and paste in ["./lists/lists_LITS"](/lists/lists_LITS) and renamed them "train.txt" and "test_vol.txt" respectively.

The dice score (%) results on testing of each folding are possible to see it in the next table.

| Folding | Liver | Tumor|
| ------ | ------ |------|
|1|    94.29%    | 53.70%|
|2|     94.52%|61.65%|
|3|93.87%|43.63%|
|4| 94.54%| 58.84%|
|5| 94.16%|56.93%|

### Ablation Study

In addition, to see how the model behave in cases where the tumor was big, and in cases where the tumor was relatively small. The model was trained and test individually just "Big Tumors" and "Small Tumors". Where big tumors is set in cases where the total quantity of pixels of the tumor was greater than 7000 and small tumors where the total number of pixels of the tumor is less than 7000. 

In the folder ["./lists/lists_LITS/train/Tumor Size"](/lists/lists_LITS/Train/Tumor size) and ["./lists/lists_LITS/test/Tumor Size"](/lists/lists_LITS/Test/Tumor Size) there are the the files for big and small tumors. To train each type of tumor individually, the train and test file has to be copy and paste in ["./lists/lists_LITS"](/lists/lists_LITS) and renamed them "train.txt" and "test_vol.txt" respectively.

The dice score (%) results on testing for each type of tumor are possible to see it in the next table.

| Type of Tumor|Tumor – Total of pixels | Liver | Tumor|
| ------ | ------ |------|------|
|Big Tumor|>7000| 92.48%   | 74.50%|
|Small| <7000|95.42|45.89%|

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)

going. You can also make an explicit request for maintainers.
