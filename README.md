# Aneja-Lab-Public-CapsNet

This repository contains the codes for training and testing 3D capsule networks for segmenting brain structures (such as thalamus and hippocampus)
on brain magnetic resonance (MR) images. For comparison purposes, we also coded 3D U-Net model as well as training and testing paradigms for U-Net. 

We have described the dataset, our models, and our results in the paper "3D Capsule Networks for Brain MRI Segmentation". The pre-print can be accessed at:
https://doi.org/10.1101/2022.01.18.22269482.


Files description:

subjects_train_valide_test_split: splits the patients in the ADNI dataset into training, validation, and test sets.

data_loader: data loader to load brain MR images from ADNI dataset.

capsnet_model: 3D capsule network model for brain MRI segmentation.

capsnet_train: code to train 3D capsule network.

capsnet_test: code to test 3D capsule network.

unet_model: 3D U-Net model for brain MRI segmentation.

unet_train: code to train 3D U-Net.

unet_test: code to test 3D U-Net.

loss_functions: repository of various loss functions that can be used for image segmentation. 



Author:
Arman Avesta, MD

Mentors:
Sanjay Aneja, MD,
Harlan Krumholz, MD, SM, 
James Duncan, PhD,
John Lafferty, PhD
