import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import pandas as pd
import io  

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
st.write("Upload images or use the camera to classify waste as Recyclable or Organic.")

# Sidebar for model metrics
with st.sidebar.expander("Model Information"):
    st.write("Model Version: 1.0")
    st.write("Last Updated: 2024-02-07")
    st.write("Training Accuracy: 95%")
    st.write("Supported Image Types: JPG, PNG, JPEG")

# Ensure session state for camera usage
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False  # Camera should be hidden initially
if "camera_image_data" not in st.session_state:
    st.session_state.camera_image_data = None  # Store captured image

# File uploader (appears first)
uploaded_files = st.file_uploader("📂 Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Button to open camera (below the file uploader)
if st.button("📷 Take Photo"):
    st.session_state.camera_open = True  # Set flag to open camera

# Show camera input only if button was clicked
camera_image = None
if st.session_state.camera_open:
    camera_image = st.camera_input("Capture Image")

# Store captured image in session state
if camera_image is not None:
    st.session_state.camera_image_data = camera_image.getvalue()
    st.session_state.camera_open = False  # Hide camera after taking photo
    uploaded_files = None  # Remove uploaded files when camera image is taken

# If a camera image exists, display it and make a prediction
if st.session_state.camera_image_data:
    st.write("### Captured Image and Prediction")
    
    # Convert stored bytes into an image
    image = Image.open(io.BytesIO(st.session_state.camera_image_data))
    st.image(image, caption="Captured Image", use_container_width=True)

    # Convert image to NumPy array
    img_array = np.array(image)

    # Convert grayscale images to RGB (if needed)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    # Resize image
    img_resized = cv2.resize(img_array, (224, 224))

    # Normalize and expand dimensions
    img_resized = img_resized / 255.0  # Normalize
    img_reshaped = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_reshaped)[0][0]  # Extract probability

    # Classification based on probability
    if prediction < 0.5:
        predicted_class = "Organic"
        confidence = (1 - prediction) * 100  # Confidence in Organic class
    else:
        predicted_class = "Recyclable"
        confidence = prediction * 100  # Confidence in Recyclable class

    # Display prediction below the camera image
    st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")

    # Save to history
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        'image': "Camera Input",
        'prediction': predicted_class,
        'confidence': confidence,
        'timestamp': datetime.now()
    })

# Display uploaded images **only if no camera image is taken**
elif uploaded_files:
    st.write("### Uploaded Images and Predictions")

    # If multiple images, arrange them in two columns
    if len(uploaded_files) > 1:
        col_1, col_2 = st.columns(2)
        columns = [col_1, col_2]
    else:
        columns = [st]

    for i, uploaded_file in enumerate(uploaded_files):
        col = columns[i % 2]  # Assign images to columns in alternation

        # Read image
        image = Image.open(uploaded_file)
        col.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to NumPy array
        img_array = np.array(image)

        # Convert grayscale images to RGB (if needed)
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)

        # Resize image
        img_resized = cv2.resize(img_array, (224, 224))

        # Normalize and expand dimensions
        img_resized = img_resized / 255.0  # Normalize
        img_reshaped = np.expand_dims(img_resized, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_reshaped)[0][0]  # Extract probability

        # Classification based on probability
        if prediction < 0.5:
            predicted_class = "Organic"
            confidence = (1 - prediction) * 100  # Confidence in Organic class
        else:
            predicted_class = "Recyclable"
            confidence = prediction * 100  # Confidence in Recyclable class

        # Display prediction below the uploaded image
        col.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")

        # Save to history
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            'image': uploaded_file.name,
            'prediction': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

# Display history
if st.checkbox("Show History"):
    if 'history' in st.session_state:
        st.write(pd.DataFrame(st.session_state.history))
    else:
        st.write("No history available.")

# Export results as CSV
if 'history' in st.session_state and st.button("Export Results"):
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        "waste_classification_results.csv",
        "text/csv"
    )
