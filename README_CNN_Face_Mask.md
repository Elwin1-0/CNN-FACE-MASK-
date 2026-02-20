# 😷 Face Mask Detection using CNN

A deep learning model that detects whether a person is wearing a face mask or not, using a Convolutional Neural Network built with TensorFlow/Keras.

## Overview

This project builds a binary image classifier that takes a face image as input and predicts **"Mask"** or **"No Mask"**. It includes a **Gradio web interface** for real-time predictions.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Gradio (Web UI)

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D + BatchNorm + MaxPool | 32 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 64 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 128 filters, 3×3 kernel |
| Flatten → Dense(128) → Dropout(0.5) | Fully connected classifier |
| Dense(1, sigmoid) | Binary output |

- **Input**: 128×128 RGB images
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy

## Dataset

- Binary classification: **With Mask** vs **Without Mask**
- Images loaded using `ImageDataGenerator` with real-time augmentation (rotation, zoom, flips, shifts)
- Train/Test split via directory structure

## Results

- Training improves steadily across epochs with good convergence
- Model generalizes well to unseen test images

## How to Run

```bash
pip install tensorflow gradio numpy pillow
jupyter notebook "CNN FACE MASK 2 (1).ipynb"
```

The Gradio interface launches at `http://127.0.0.1:7860` — upload any face image to get a prediction.

## Project Structure

```
├── CNN FACE MASK 2 (1).ipynb   # Main notebook with model + Gradio app
└── README.md
```
