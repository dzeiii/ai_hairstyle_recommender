import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Set up page styling
st.set_page_config(page_title="HAIR WE GO!", page_icon="💇‍♂️", layout="centered")

st.title("💇‍♂️ HAIR WE GO!")
st.subheader("AI Hairstyle Recommendation Dashboard")

# 1. Load your trained AI model
@st.cache_resource
def load_face_model():
    # Returns your trained model file
    return tf.keras.models.load_model('face_shape_model.h5')

try:
    model = load_face_model()
    # Ensure these labels perfectly match your Kaggle training folder names alphabetically
    LABELS = ['heart', 'oblong', 'oval', 'round', 'square']
except Exception as e:
    st.error("Model file 'face_shape_model.h5' not found. Please train your model first!")
    st.stop()

# 2. Sidebar options for user settings
st.sidebar.header("User Settings")
gender = st.sidebar.radio("Select Gender Category:", ("men", "women"))

st.write("---")
st.write("### 📸 Step 1: Capture or Upload Your Face Image")

# 3. Handle image intake from the user's camera stream or file upload
source_option = st.radio("Choose Input Method:", ("Use Live Camera", "Upload Image File"))

captured_image = None
if source_option == "Use Live Camera":
    camera_input = st.camera_input("Position your face clearly in front of the lens")
    if camera_input is not None:
        captured_image = Image.open(camera_input)
else:
    file_input = st.file_uploader("Upload a front-facing portrait photo", type=["jpg", "jpeg", "png"])
    if file_input is not None:
        captured_image = Image.open(file_input)

# 4. Process and predict when an image is received
if captured_image is not None:
    st.write("---")
    st.write("### 🧠 Step 2: Processing AI Classification...")
    
    # Render the input photo safely to the UI layout
    st.image(captured_image, caption="Analyzed Portrait", width=250)
    
    # Format picture matrix dimensions into array shapes required by MobileNetV2
    img_array = np.array(captured_image.convert('RGB'))
    resized_img = cv2.resize(img_array, (224, 224))
    normalized_img = resized_img / 255.0
    input_batch = np.expand_transform_dims = np.expand_dims(normalized_img, axis=0)
    
    # Run prediction weights logic 
    predictions = model.predict(input_batch, verbose=0)
    highest_score_index = np.argmax(predictions)
    detected_shape = LABELS[highest_score_index]
    confidence_score = predictions[0][highest_score_index] * 100
    
    # Display AI evaluation summary metric
    st.success(f"**Analysis Complete!** We detected a **{detected_shape.upper()}** face shape (Confidence: {confidence_score:.1f}%)")
    
    st.write("---")
    st.write(f"### 📋 Step 3: Your Suggested {gender.capitalize()} Hairstyles")
    
    # 5. Dynamically fetch and present your specific asset path matching the user's criteria
    recommendation_path = f"hairstyle_dataset/{detected_shape}/{gender}.png"
    
    try:
        recommendation_graphic = Image.open(recommendation_path)
        st.image(recommendation_graphic, use_container_width=True, caption=f"Best styles for {detected_shape.upper()} faces ({gender.upper()})")
    except FileNotFoundError:
        st.error(f"Asset file missing: Could not locate '{recommendation_path}' in directory folders.")
