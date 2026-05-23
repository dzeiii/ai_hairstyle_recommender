import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
import os

# Set up page layout
st.set_page_config(page_title="HAIR WE GO!", page_icon="💇‍♂️", layout="centered")

st.title("💇‍♂️ HAIR WE GO!")
st.subheader("AI Hairstyle Recommendation Dashboard")

# Handle truncated images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CLEAN LOCAL PATHS ---
MODEL_PATH = 'face_shape_model.h5'
ASSET_DIR = 'hairstyle_dataset'

@st.cache_resource
def load_face_model():
    return tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    model = load_face_model()
    LABELS = ['heart','oval', 'round', 'square']
else:
    st.error("Model file 'face_shape_model.h5' not found in your folder!")
    st.stop()

# Sidebar Setup
st.sidebar.header("User Settings")
gender = st.sidebar.radio("Select Gender Category:", ("men", "women"))

st.write("---")
st.write("### 📸 Step 1: Capture or Upload Your Face Image")
source_option = st.radio("Choose Input Method:", ("Use Live Camera", "Upload Image File"))

captured_image = None
if source_option == "Use Live Camera":
    camera_input = st.camera_input("Position your face clearly in the center")
    if camera_input is not None:
        captured_image = Image.open(camera_input)
else:
    file_input = st.file_uploader("Upload a front-facing portrait photo", type=["jpg", "jpeg", "png"])
    if file_input is not None:
        captured_image = Image.open(file_input)

# Processing Logic
if captured_image is not None:
    st.write("---")
    st.write("### 🧠 Step 2: Processing AI Classification...")
    st.image(captured_image, width=250, caption="Analyzed Face Profile")
    
    # Process for MobileNetV2
    img_array = np.array(captured_image.convert('RGB'))
    resized_img = cv2.resize(img_array, (224, 224)) / 255.0
    input_batch = np.expand_dims(resized_img, axis=0)
    
        # Run prediction
    predictions = model.predict(input_batch, verbose=0)[0]  # <-- ADDED [0] HERE
    
    highest_score_index = np.argmax(predictions)
    detected_shape = LABELS[highest_score_index]
    confidence_score = predictions[highest_score_index] * 100
    
    st.success(f"🎉 **Analysis Complete!** We detected a **{detected_shape.upper()}** face shape ({confidence_score:.1f}%)")
    
    st.write("---")
    st.write(f"### 📋 Step 3: Your Suggested {gender.capitalize()} Hairstyles")
    
    recommendation_path = os.path.join(ASSET_DIR, f"{detected_shape}/{gender}.png")
    
    if os.path.exists(recommendation_path):
        recommendation_graphic = Image.open(recommendation_path)
        st.image(recommendation_graphic, use_container_width=True)
    else:
        st.error(f"Asset missing at: {recommendation_path}")
