import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your dataset
train_images_path = r'C:\Users\91741\Downloads\Implement\train_images\train_images'
train_masks_path = r'C:\Users\91741\Downloads\Implement\train_masks\train_masks'

def load_data(images_path, masks_path, limit_per_class=500):
    images = []
    masks = []
    tumor_images = 0
    non_tumor_images = 0

    # List of all mask files
    mask_files = sorted(os.listdir(masks_path))
    for mask_file in mask_files:
        # Determine if this mask is tumorous
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        has_tumor = np.any(mask > 0)

        # Collect tumorous and non-tumorous images up to the limit
        if has_tumor and tumor_images < limit_per_class:
            image_path = os.path.join(images_path, mask_file.replace('mask', 'image'))
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for faster training
                images.append(img)
                masks.append(cv2.resize(mask, (128, 128)))
                tumor_images += 1
        elif not has_tumor and non_tumor_images < limit_per_class:
            image_path = os.path.join(images_path, mask_file.replace('mask', 'image'))
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize for faster training
                images.append(img)
                masks.append(cv2.resize(mask, (128, 128)))
                non_tumor_images += 1

        # Stop if we have enough images of both types
        if tumor_images >= limit_per_class and non_tumor_images >= limit_per_class:
            break

    images = np.array(images) / 255.0  # Normalize images
    masks = np.array(masks) / 255.0  # Normalize masks
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension

    return np.array(images), np.array(masks)

# Load balanced dataset of 1000 images (500 tumorous, 500 non-tumorous)
images, masks = load_data(train_images_path, train_masks_path, limit_per_class=500)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# U-Net model
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    concat1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(concat2)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the U-Net model
model = unet_model()

# Early stopping to stop training if it takes too long
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model with data augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=25,  # Increased epochs for more comprehensive training
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('liver_cancer_unet_balanced.h5')
