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
st.title("🚮 Waste Classification Using CNN")
st.write("Upload images, use the camera, or check history for waste classification.")


with st.sidebar:
    st.markdown("""
        <style>
        .css-1d391kg {width: 300px !important;}
        </style>
        """, unsafe_allow_html=True)

    st.title("ℹ️ Information")
    
    with st.sidebar.expander("Model Information"):
        st.write("Model Version: 1.0")
        st.write("Last Updated: 2024-02-07")
        st.write("Training Accuracy: 92.47%")
        st.write("Supported Image Types: JPG, PNG, JPEG")
        
    with st.sidebar.expander("Class Labels"):
        st.write("🌱 Organic ") 
        st.write("♻️ Recyclable")
    
    with st.sidebar.expander("How to Use"):
        st.write("1. Upload an image or use the camera.")
        st.write("2. The model will predict whether the image is organic or recyclable.")
        st.write("3. View the prediction history.")            
        
    with st.sidebar.expander("About"):
        st.write("This is a simple web app to classify waste images into organic and recyclable categories.")
        st.write("It uses a Convolutional Neural Network (CNN) model trained on a dataset of waste images.")
        st.markdown("The model has been trained on [dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data/data), and achieves an accuracy of 92% on the test set.")
        st.write("You can upload images, use the camera, and view the prediction history using the tabs below.")

    st.markdown("Made by [Jayendra](https://www.linkedin.com/in/voutla-jayendra-553abb313/)")
    
# Ensure session state for camera usage
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False
if "camera_image_data" not in st.session_state:
    st.session_state.camera_image_data = None
if "history" not in st.session_state:
    st.session_state.history = []

# Tabs for different functionalities
tabs = ["📂 Upload Images", "📷 Camera Input", "📜 History"]
selected_tab = st.tabs(tabs)

# File Upload Tab
with selected_tab[0]:
    st.header("Upload Images")
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("### Uploaded Images and Predictions")
        cols = st.columns(2) if len(uploaded_files) > 1 else [st]
        
        for i, uploaded_file in enumerate(reversed(uploaded_files)):
            col = cols[i % 2]
            image = Image.open(uploaded_file)
            image.thumbnail((200, 200))  # Dynamically reduce size
            col.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process and Predict
            img_array = np.array(image)
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            img_resized = cv2.resize(img_array, (224, 224)) / 255.0
            img_reshaped = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_reshaped)[0][0]
            predicted_class = "Organic" if prediction < 0.5 else "Recyclable"
            confidence = (1 - prediction if prediction < 0.5 else prediction) * 100
            
            col.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")
            st.session_state.history.insert(0, {'image': uploaded_file.name, 'prediction': predicted_class, 'confidence': confidence, 'timestamp': datetime.now()})

# Camera Tab
with selected_tab[1]:
    st.header("Camera Input")
    if st.button("📷 Open Camera" if not st.session_state.camera_open else "❌ Close Camera"):
        st.session_state.camera_open = not st.session_state.camera_open
    
    if st.session_state.camera_open:
        camera_image = st.camera_input("Capture Image")
        if camera_image:
            st.session_state.camera_image_data = camera_image.getvalue()
            st.session_state.camera_open = False

    if st.session_state.camera_image_data:
        st.write("### Captured Image and Prediction")
        image = Image.open(io.BytesIO(st.session_state.camera_image_data))
        st.image(image, caption="Captured Image", use_container_width=True)
        
        img_array = np.array(image)
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_reshaped)[0][0]
        predicted_class = "Organic" if prediction < 0.5 else "Recyclable"
        confidence = (1 - prediction if prediction < 0.5 else prediction) * 100
        
        st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence")
        st.session_state.history.insert(0, {'image': "Camera Input", 'prediction': predicted_class, 'confidence': confidence, 'timestamp': datetime.now()})

# History Tab
with selected_tab[2]:
    st.header("Prediction History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width=900)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "waste_classification_results.csv", "text/csv")
    else:
        st.write("No history available.")
