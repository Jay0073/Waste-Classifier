import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classification_model.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = load_model()

# Define class labels
CLASS_NAMES = ["Organic", "Recyclable"]

# Streamlit UI
st.title("♻️ Waste Classification AI")
st.write("Upload an image to classify it as Recyclable or Organic.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to NumPy array
    img_array = np.array(image)

    # Convert grayscale images to RGB (if needed)
    if img_array.ndim == 2:  
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Ensure 3 color channels
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    # Resize image
    img_resized = cv2.resize(img_array, (224, 224))  

    # Normalize and expand dimensions
    img_resized = img_resized / 255.0  # Normalize
    img_reshaped = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_reshaped)[0][0]  # Extract single probability value
    
    # Classification based on probability
    if prediction < 0.5:
        predicted_class = "Organic"
        confidence = (1 - prediction) * 100  # Confidence in Organic class
    else:
        predicted_class = "Recyclable"
        confidence = prediction * 100  # Confidence in Recyclable class

    # Display result
    st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")
