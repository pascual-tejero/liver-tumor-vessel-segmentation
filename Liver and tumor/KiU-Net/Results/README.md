## Results

This folder contains subfolders for binary and multi-class segmentation tasks, each containing further subfolders for different architecture and loss combinations used to obtain corresponding results.

Each subfolder for both tasks contains the following items:

- A file named _Summary.txt_, which displays the neural network utilized and its respective hyperparameters.
- Patient-wise predictions for the test dataset saved in the .nii format.
- An Excel file named _results.xlsx_ which contains the Dice score for each patient in the test dataset, calculated per class label and globally. This file also includes the mean, standard deviation, minimum, and maximum Dice score per class label.

For the multi-class segmentation task, two additional subfolders exist, namely Cross-validation and Ablation study using KiU-Net architecture. These subfolders use different configurations of train and test sets as follows:

- In the Cross-validation subfolder, we have used a 5-fold cross-validation technique. Here, the KiU-Net model was trained and evaluated five times, each time with a different test set.
- In the Ablation study subfolder, we divided the dataset into two subsets (big and small tumor) based on the number of tumor labels: high tumor label representation (> 7000 tumor labels in a patient) and low tumor label representation (< 7000 tumor labels in a patient). We obtained 56 and 62 datasets, respectively, and discarded 13 patients that did not have tumor segmentation in the mask. We then divided each subset into train and test sets and ran the KiU-Net model, achieving better results for the high tumor label representation dataset than the other.

Here, there's an example of the patient 27 of the CT scan, the ground truth mask and the segmentation by KiU-Net:
<table>
  <tr>
    <td><img src="../img/ct_pt27.png" alt="CT image"></td>
    <td><img src="../img/gt_pt27.png" alt="Ground truth segmentation"></td>
    <td><img src="../img/pred_pt27.png" alt="Predicted segmentation"></td>
  </tr>
</table>

