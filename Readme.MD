## Overview
To ensure the privacy of individuals visible in the India Driving Dataset (IDD), we applied a face anonymization technique by replacing real faces with generated fake images. This process protects user privacy while maintaining the usability of the dataset for research purposes.

## Objective
The goal of this evaluation is to verify that the face anonymization process does not significantly alter the semantic structure of the images, ensuring that the dataset remains useful for tasks like semantic segmentation while preserving privacy.

## Quality Evaluation of Anonymized Faces
To evaluate the quality of the generated fake faces, we performed the following steps:

- **Semantic Segmentation on Original and Anonymized Images**:
Semantic segmentation was applied to both the original images and the corresponding anonymized images using the same segmentation model.

- **Mask Comparison**:
The segmentation masks obtained from the original images (original mask) were compared with the masks from the anonymized images (fake mask).

- **Metric Calculation**:
The similarity between the original and fake masks was quantified using a metric such as Mean Intersection over Union (mIoU) and Dice Coefficient.

## Models
- [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- [FCN (Fully Convolutional Network)](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)

## Directory Structure
Each directory in the project contains two files:

1. `mask.py`

This script generates semantic segmentation masks for the images in the dataset.

2. `metrics.py`

This script calculates evaluation metrics for the segmentation masks.


## Results
- **DeepLabV3 (Resnet101)**: mIoU = **97.61**, Dice: **99.89**
- **FCN (Resnet101)**: mIoU = **98.06**, Dice: **99.91**