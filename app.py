import streamlit as st
import tensorflow as tf
import numpy as np
import urllib.request
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="AI Hairstyle Recommender", page_icon="✂️", layout="centered")

# 1. Download Model from Google Drive dynamically if it doesn't exist locally
@st.cache_resource
def load_my_model():
    MODEL_PATH = 'hairstyle_model.h5'
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model from Google Drive... Please wait a moment."):
            FILE_ID = "1BrcUSYWTHb4F2Oz3h_wapVqiQadXZhpC" 
            URL = f"https://google.com{FILE_ID}"
            urllib.request.urlretrieve(URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. User Interface Content
st.title("AI Hairstyle Recommender ✂️")
st.write("Upload a clear front-facing photo of yourself to find your face shape and the best matching hairstyles!")

uploaded_file = st.file_uploader("Choose a selfie photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Selfie', use_container_width=True)
    
    st.info("🔄 Processing image and scanning facial structure...")
    
    img = image.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"### Target Match Identified: **{predicted_class}** ({confidence:.1f}% Confidence)")
    
    st.write("---")
    st.write("### 💇‍♂️ Recommended Hairstyles for Your Face Shape:")
    
    if predicted_class == 'Heart':
        st.markdown("- **Side-Swept Fringes:** Softens a wide forehead.\n- **Long Layers:** Adds volume around the lower jawline.\n- **Textured Pixie Cut:** Enhances natural bone structure features.")
    elif predicted_class == 'Oblong':
        st.markdown("- **Side Partings:** Visually widens the facial structure look.\n- **Fringe/Bangs:** Cuts down on vertical length perception.\n- **Voluminous Curls:** Adds balancing width to side profiles.")
    elif predicted_class == 'Oval':
        st.markdown("- **Blunt Bob:** Complements balanced structural symmetries.\n- **Slicked Back Undercut:** Clean style showcasing smooth contours.\n- **Long Waves:** Universally framing for oval configurations.")
    elif predicted_class == 'Round':
        st.markdown("- **Pompadour:** Adds flattering height to vertical features.\n- **Asymmetrical Cuts:** Breaks up rounder circular symmetries.\n- **Long Layered Shag:** Creates structural angles along cheeks.")
    elif predicted_class == 'Square':
        st.markdown("- **Soft Layered Waves:** Tones down sharp, boxy angles.\n- **Side Swept Bob:** Shifts visual focus away from jawlines.\n- **Buzz Cut:** Accents strong, athletic structural lines.")
