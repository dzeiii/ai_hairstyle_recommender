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
    
    # 1. Format the folder and base filename into clean lower strings
    shape_folder = detected_shape.lower().strip()
    gender_file = gender.lower().strip()
    
    # 2. List all common extensions to check sequentially
    possible_extensions = ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']
    recommendation_path = None
    
    # Loop through the list to find which version actually exists on the hosting server
    for ext in possible_extensions:
        test_path = os.path.join(ASSET_DIR, f"{shape_folder}/{gender_file}{ext}")
        if os.path.exists(test_path):
            recommendation_path = test_path
            break  # Stop checking once we find a match!

    # 3. Render the discovered file asset safely to the interface layout
    if recommendation_path is not None:
        recommendation_graphic = Image.open(recommendation_path)
        st.image(recommendation_graphic, use_container_width=True, caption=f"Best styles for {detected_shape.upper()} faces")
    else:
        # If none of the extension styles were found, display an informative error log
        st.error(f"❌ Asset file missing inside folder structure! Searched for '{gender_file}' variations inside 'hairstyle_dataset/{shape_folder}/'")

        # --- NEW FEATURE: STYLING TIPS TO AVOID MATRIX ---
        st.write("---")
        st.write("### ⚠️ Hairstyles to Avoid for Your Face Shape")
        
        # Comprehensive professional dictionary mapping rule configurations
        avoidance_tips = {
            'oval': [
                "**Avoid heavy, long straight bangs** that cut straight across your face, as they block your features and make a naturally balanced oval head shape look shorter.",
                " Avoid hairstyles that add excessive height or volume directly on the top without any width, which can make your face appear artificially long."
            ],
            'round': [
                "**Avoid sleek, chin-length bobs with flat surfaces** that hug your face line, as they act like a border highlighting the roundness of your cheeks.",
                " Avoid slicked-back styles or middle parts with no volume, which can compress your forehead proportions and make the face shape look even wider."
            ],
            'heart': [
                "**Avoid heavy top volume or slick back styles with high pompadours**, as they add bulk to an already wide forehead line and accent a narrow chin.",
                " Avoid short, blunt-cut wispy bangs or styles that end harshly right at your cheekbone levels, which can visually expand the upper half of your face."
            ],
            'square': [
                "**Avoid sharp, blunt-cut straight fringes or geometric box haircuts**, as these parallel lines emphasize harsh jawline edges and make your face look boxy.",
                " Avoid slicked-back ponytails, tightly pulled updos, or center parts with completely flat sides that offer zero soft layers around the sides of your face."
            ]
        }
        
        # Safely fetch matching tip text list based on what the AI model detected
        current_shape = detected_shape.lower().strip()
        
        if current_shape in avoidance_tips:
            # Render a styled container box to catch the user's eye
            with st.container(border=True):
                st.markdown(f"#### 🚫 Styling Red Flags for {detected_shape.upper()} Profiles:")
                for tip in avoidance_tips[current_shape]:
                    st.write(f"- {tip}")

