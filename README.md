# Unbalanced Taxonomy Instance Segmentation

This repository contains the code for a Master's thesis focusing on computer vision research, specifically targeting instance segmentation challenges. The project evaluates and enhances instance segmentation techniques on the TACO (Trash Annotation in Context) dataset using state-of-the-art computer vision models.

## Features

- **Instance Segmentation**: Apply instance segmentation using fine-tuned models like MaskRCNN and Mask2Former.
- **Zero-Shot Segmentation**: Utilize FC-CLIP for zero-shot learning capabilities in instance segmentation.
- **Dynamic Weight Balancing**: Implement dynamic weight balancing loss functions to improve model performance for underrepresented classes in the dataset.

## Installation

To set up the project, clone the repository and run the setup script:

```bash
git clone https://github.com/yourusername/unbalanced-taxonomy-instance-segmentation.git
cd unbalanced-taxonomy-instance-segmentation
./setup.sh
