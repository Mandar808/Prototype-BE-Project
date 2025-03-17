# Liver Cancer Segmentation Using U-Net

This repository contains an implementation of a U-Net-based deep learning model for segmenting liver tumors from medical images. The model is trained on a dataset containing both tumorous and non-tumorous liver images.

## Dataset

The dataset consists of:
- **train_images**: Contains liver images (both with and without tumors).
- **train_masks**: Contains corresponding segmentation masks, where tumor regions are marked.

### Data Preprocessing
- Images and masks are resized to **128x128** for faster training.
- A balanced dataset of **500 tumorous** and **500 non-tumorous** images is created.
- Images are normalized to values between 0 and 1.

## Model Architecture

The model is implemented using **U-Net**, which consists of:
- **Encoder**: Two convolutional layers followed by max pooling.
- **Bottleneck**: A convolutional layer connecting encoder and decoder.
- **Decoder**: Two upsampling layers with concatenation from the encoder.
- **Output Layer**: A single-channel convolution with sigmoid activation.

## Requirements

To run the project, install the dependencies using:
```bash
pip install tensorflow numpy opencv-python scikit-learn
```

## Training the Model

Run the following command to train the model:
```bash
python train.py
```

Training includes:
- **Data augmentation** (rotation, shifting, zoom, flipping) using `ImageDataGenerator`.
- **Early stopping** to prevent overfitting.
- Training with **25 epochs** and batch size **8**.

## Model Evaluation

The model is evaluated based on:
- **Binary cross-entropy loss**
- **Accuracy**

## Saving the Model

After training, the model is saved as:
```bash
liver_cancer_unet_balanced.h5
```


