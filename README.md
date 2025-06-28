# Sign Language Classification (Indian Sign Language)

A MATLAB-based CNN project to classify Alphabetic Characters hand gestures using image classification. The model is trained on a labeled dataset and saved as `sign_language_model.mat` for future deployment.

## Model Overview

- Input: 64x64 RGB images of hand gestures
- Architecture: 3-layer CNN with ReLU, BatchNorm, MaxPooling
- Output: Categorical labels representing ISL signs

## Files

- `sign_language_model.mat` – Trained CNN model
- `train_sign_model.m` – MATLAB training script
- `sample_images/` – Demo inputs

## Accuracy

```matlab
Validation Accuracy: %
