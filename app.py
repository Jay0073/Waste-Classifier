import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import pandas as pd

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

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Camera input
camera_input = st.camera_input("Take a picture")

if uploaded_files or camera_input:
    images = []

    if uploaded_files:
        images.extend(uploaded_files)

    if camera_input:
        images.append(camera_input)

    for uploaded_file in images:
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

        # Show detailed classification information
        def show_waste_info(predicted_class):
            if predicted_class == "Organic":
                st.info("""
                **Organic Waste Handling:**
                - Compostable ✅
                - Biodegradable time: 2-6 weeks
                - Recommended disposal: Compost bin
                """)
            else:
                st.info("""
                **Recyclable Waste Handling:**
                - Recyclable ✅
                - Processing method: Industrial recycling
                - Recommended bin: Recycling bin
                """)
                
        show_waste_info(predicted_class)

        # Add to session state history
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            'image': uploaded_file.name if uploaded_file else 'Camera Input',
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
