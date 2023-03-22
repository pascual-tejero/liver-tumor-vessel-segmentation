
# train_ct_path = 'Data/CT/LiTS/Train/Original/ct'  

# train_seg_path = '/Data/CT/LiTS/Train/Original/seg'
  

test_ct_path = 'Dataset/LiTS/Ablation study (256x256)/Big tumor dataset/test/ct'
# test_ct_path = 'Dataset/Medical Segmentation Decathlon/Multi-class segmentation/256x256/test/ct'

test_seg_path = 'Dataset/LiTS/Ablation study (256x256)/Big tumor dataset/test/seg'  
# test_seg_path = 'Dataset/Medical Segmentation Decathlon/Multi-class segmentation/256x256/test/seg'

training_set_path = 'Dataset/LiTS/Ablation study (256x256)/Big tumor dataset/train'
# training_set_path = 'Dataset/Medical Segmentation Decathlon/Multi-class segmentation/256x256/train'

pred_path = 'Results/Multi-class segmentation/15_AblationStudy_48slices_CrossEntropyLoss_256x256_ResUNet/Big Tumor/'

crf_path = './crf' 
 
module_path = 'Results/Multi-class segmentation/15_AblationStudy_48slices_CrossEntropyLoss_256x256_ResUNet/Big Tumor/...'

size = 48  # Using x consecutive slices as input to the network

down_scale = 0.5  # Cross-sectional downsampling factor

expand_slice = 20  # Only 20 slices containing the liver and the top and bottom of the liver were used as training samples

slice_thickness = 1  # Normalise the spacing of all data in the z-axis to 1mm

upper, lower = 200, -200  # CT data grey scale truncation window

# ---------------------Training data to obtain relevant parameters-----------------------------------


# -----------------------Network structure related parameters------------------------------------

drop_rate = 0.3  # dropout random dropout probability

# -----------------------Network structure related parameters------------------------------------


# ---------------------Network training related parameters--------------------------------------

gpu = '0'  # Serial number oef the graphics card used

Epoch = 5501

learning_rate = 1e-4

learning_rate_decay = [500, 700]

alpha = 0.33  # Depth of supervision attenuation factor

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------Network training related parameters--------------------------------------


# ----------------------Model test related parameters-------------------------------------

threshold = 0.5  # Threshold Degree

stride = 12  # Sliding sampling step

maximum_hole = 5e4  # Largest cavity area

# ----------------------Model test related parameters-------------------------------------


# ---------------------CRF post-processing optimisation related parameters----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # Number of extensions in three directions based on predicted results

max_iter = 20  # Number of CRF iterations

s1, s2, s3 = 1, 10, 10  # CRF Gaussian kernel parameters

# ---------------------CRF post-processing optimisation related parameters----------------------------------