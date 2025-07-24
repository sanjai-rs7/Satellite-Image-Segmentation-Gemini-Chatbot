# Satellite Image Segmentation with InceptionResNetV2‑UNet

A modular deep learning system for multi-class semantic segmentation of satellite imagery using a custom InceptionResNetV2‑UNet architecture. This project includes data preprocessing, model training, performance benchmarking, a RESTful API, and an interactive multimodal chatbot interface for real-time inference and explanation.

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)


## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Architecture](#architecture)  
- [Training Details](#training-details)  
- [Evaluation](#evaluation)  
- [Chatbot Interface](#chatbot-interface)  
- [Deployment](#deployment)  
- [API Endpoints](#api-endpoints)  
- [Setup Instructions](#setup-instructions)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Overview

This project addresses the task of satellite image segmentation across six land-use classes: **Water, Land, Road, Building, Vegetation**, and **Unlabeled**. A custom InceptionResNetV2‑UNet was developed and trained using a hybrid loss function, achieving high accuracy and generalization on real-world aerial data. The solution supports end-to-end inference and explanation via an interactive chatbot interface powered by both image and text inputs.

---

## Dataset

- **Source**: Humans in the Loop (HiTL) semantic segmentation dataset  
- **Classes**: Water, Land, Road, Building, Vegetation, Unlabeled  
- **Preprocessing**:
  - Dataset unified from 8 separate “Tile” folders
  - 576 total images: 460 for training and 116 for validation
  - Masks converted to 6-channel softmax-compatible format
- **Augmentations** (via Albumentations):
  - Random cropping (256x256)
  - Horizontal/vertical flips
  - Rotations (60°–300°)
  - CLAHE
  - Grid and optical distortions
  - Brightness/contrast adjustments

---

## Architecture

### InceptionResNetV2‑UNet

- **Encoder**: Pretrained InceptionResNetV2 (ImageNet)
- **Bridge**: Dropout + bottleneck layers
- **Decoder**: 4 transposed convolution blocks with skip connections
- **Output**: 6-channel softmax segmentation mask

### Loss Function

- Combined **Categorical Cross-Entropy** and **Dice Loss** to optimize both per-pixel accuracy and region-level overlap quality

---

## Training Details

- **Epochs**: 50  
- **Batch Size**: 16  
- **Optimizer**: Adam (`lr=1e-4`, reduced to `2e-5` via `ReduceLROnPlateau`)  
- **Callbacks**: EarlyStopping (patience=10), ModelCheckpoint  
- **Input Shape**: 256x256 RGB  
- **Total Parameters**: ~55M (trainable subset fine-tuned)

---

## Evaluation

### Validation Performance

| Metric             | Value at Epoch 42 |
|--------------------|-------------------|
| Dice Coefficient   | 0.678             |
| Accuracy           | 87.17%            |
| Training Dice      | 0.611             |
| Training Accuracy  | 78.2%             |

### Class-wise IoU Comparison (Low-Res vs High-Res)

| Class      | Low-Res IoU | High-Res IoU | Improvement |
|------------|-------------|--------------|-------------|
| Road       | 0.62        | 0.81         | +19%        |
| Building   | 0.67        | 0.85         | +18%        |
| Vegetation | 0.76        | 0.83         | +7%         |

**Note:** Visual overlays and boundary F1-scores confirm qualitative improvements on small structures.

---

## Chatbot Interface

A multimodal chatbot was developed to support real-time inference and analysis via both image and text inputs.

- **Capabilities**:
  - Accepts user-uploaded satellite images
  - Accepts natural language queries
  - Returns segmentation masks, class-level pixel distributions, and interpretability context
- **Backend**: Vision-Language model integrated with REST API
- **Frontend**: Available for interaction via HTTP clients or frontend dashboard

---


