[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kbnzRo6k)
# Homework 4: CNNs vs Fully Connected Networks (FCNs)

## Overview

In this homework, you will explore the differences between Convolutional Neural Networks (CNNs) 
and Fully Connected Networks (FCNs), also known as Multilayer Perceptrons or MLPs, 
using medical image datasets.
You will:

- Load and process image data using custom and prebuilt PyTorch transforms.
- Understand the architectural differences between CNNs and FCNs.
- Implement and train both CNN and FCN models for classification tasks.
- Visualize intermediate activations and output predictions.
- Compare how each architecture handles spatial information and generalizes to unseen data.

---

## Datasets

You will work with one primary dataset:
1. Chest X-Ray (Binary Classification)
- **Images**: Frontal chest radiographs
- **Labels**: Normal or pneumonia
- **Task**: Predict presence of pneumonia in a chest X-ray

The dataset will be loaded using the provided `HW4DataLoader` class.
It will be split into separate CSV files for training, validation, and testing.

---

## Installation

Install dependencies using pip:

1. **Clone** this repo (first time only):
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework4
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework4
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw4
    ```
3. Install dependencies:
    ```bash
    source .hw4/bin/activate (Linux/MacOS) or .\.hw4\Scripts\activate
    pip install -r requirements.txt
    ```

---

## Preprocessing and Augmentation

You will experiment with different image preprocessing and augmentation techniques:

- Padding, cropping, resizing
- Tensor conversion and channel manipulation
- Custom transform pipelines for training vs inference

These help prepare data for CNN or FCN processing and influence how models generalize.

---

## Tasks

You will complete the following:

### Data + Exploration
- [ ] Load the dataset and confirm split sizes (train/val/test)
- [ ] Visualize example images from each class
- [ ] Plot class distributions
- [ ] Implement transform pipelines:
  - [ ] training transforms (with augmentation)
  - [ ] evaluation transforms (no augmentation)

### A) FCN baseline 
- [ ] Build an FCN that classifies images after flattening
- [ ] Track input/output tensor shapes carefully
- [ ] Train and evaluate it on the dataset

### B) CNN classifier
- [ ] Build a CNN that preserves spatial structure
- [ ] Print intermediate tensor shapes through the forward pass
- [ ] Train and evaluate it on the same splits as the FCN classifier

### Evaluation + Analysis

For both models (FCN and CNN), report:
- [ ] Accuracy on val/test
- [ ] AUROC on val/test
- [ ] Confusion matrix on test
- [ ] Training curves (loss + metric)

Visualization + interpretation:
- [ ] Visualize CNN intermediate feature maps for a few examples
- [ ] Compare how FCN vs CNN responds to augmentation and distribution shift
- [ ] Explain what spatial information is lost when flattening an image

### Second Dataset
- [ ] Repeat the pipeline on a second image dataset of your choice (biomedical is strongly recommended). You can look for datasets on Kaggle or Hugging Face.

---

## Final Reflection

You will then write a **2–3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated.

- [ ] How did performance differ between the FCN and CNN?
- [ ] Why do CNNs typically perform better for image classification?
- [ ] What spatial information was preserved or lost in each architecture?
- [ ] What did you learn from visualizing intermediate layers?
- [ ] Which architecture was more robust to augmentation and why?

---

## Expected Skills

By the end of this homework, you should be able to:

- Implement both FCN and CNN models for image classification.
- Evaluate classification models using appropriate metrics.
- Visualize feature maps and interpret spatial reasoning in CNNs.
- Understand how convolutional architectures encode information.
- Understand the limitations of flattening image inputs for dense networks.
