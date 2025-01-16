import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Dice loss function definition
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

# Load the trained U-Net model with dice_loss as a custom object
model = load_model('liver_cancer_unet_balanced.h5', custom_objects={'dice_loss': dice_loss})

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to detect tumor presence and visualize prediction
def detect_tumor(prediction, threshold=0.1):
    mask = (prediction[0, :, :, 0] > threshold).astype(np.uint8)
    tumor_detected = np.any(mask)
    return tumor_detected, mask

# Streamlit app
st.title("Liver Cancer Detection Using U-Net")
st.write("Upload a liver scan image to check for tumor detection.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    # Preprocess and predict
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    
    # Detect tumor presence and display mask
    tumor_detected, mask = detect_tumor(prediction)

    # Display the prediction result
    if tumor_detected:
        st.write("Tumor Detected!")
    else:
        st.write("No Tumor Detected.")
    
    # Plot the original image and predicted mask for visual debugging
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Predicted Mask")
    ax[1].axis('off')
    
    st.pyplot(fig)
