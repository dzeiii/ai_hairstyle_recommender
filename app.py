import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="AI Hairstyle Matrix Recommender", page_icon="✂️", layout="wide")

# 1. Pull Trained Weights Securely from Google Drive
@st.cache_resource
def load_my_model():
    MODEL_PATH = 'hairstyle_model.h5'
    if not os.path.exists(MODEL_PATH):
        FILE_ID = "1BrcUSYWTHb4F2Oz3h_wapVqiQadXZhpC" 
        URL = f"https://google.com{FILE_ID}"
        urllib.request.urlretrieve(URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()
CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# 2. Hardcoded Filter Overlays Mapping Rules
HAIRSTYLES_LABELS = {
    'Heart':  ["Side Fringe", "Long Layers", "Pixie Cut", "Wispy Bangs", "Bob Cut", "Taper Fade", "Textured Crop", "Curtain Bangs", "Classic Pompadour"],
    'Oblong': ["Side Part", "Blunt Bangs", "Voluminous Curls", "Shag Cut", "Chin Bob", "Wavy Lob", "Slick Back", "Buzz Cut", "Modern Quiff"],
    'Oval':   ["Blunt Bob", "Slick Undercut", "Long Waves", "French Crop", "Crew Cut", "Layered Shag", "Curtain Hair", "Side Swept", "Fringe Crop"],
    'Round':  ["High Pompadour", "Asymmetric Bob", "Long Shag", "Spiky Texture", "Undercut Fade", "Side Bangs", "Faux Hawk", "Textured Quiff", "Ivy League"],
    'Square': ["Layered Waves", "Side Swept Bob", "Clean Buzz", "Crew Cut", "Textured Crop", "Classic Part", "Comb Over", "Long Shag", "Slicked Back"]
}

# 3. User Interface Layout Architecture
st.title("AI Hairstyle Filter Matrix Recommender ✂️")
st.write("Capture a snapshot using your webcam. The AI will classify your facial structure and display your look under 9 distinct style filters inside a 3x3 layout grid!")

# Use Streamlit's native, lightweight camera capture feature
cam_image = st.camera_input("Smile for the camera!")

if cam_image is not None:
    # Convert image format for processing
    image = Image.open(cam_image)
    img_cv = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # EXACT COORD CENTER MATRIX CROP FIX
    h, w, _ = img_cv.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_face = img_cv[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Process for Model Inference
    with st.spinner("AI is analyzing your face structure contour points..."):
        resized = cv2.resize(cropped_face, (224, 224))
        normalized = resized / 255.0
        tensor_input = np.expand_dims(normalized, axis=0)
        preds = model.predict(tensor_input, verbose=0)
        predicted_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.success(f"### Detected Face Shape: **{predicted_class}** ({confidence:.1f}% Match Confidence)")
    st.write("### Your Custom 3x3 Hairstyle Try-On Matrix:")
    
    # 4. Generate the 3x3 Filter Matrix Image
    sub_size = 300
    grid_img = np.zeros((sub_size * 3, sub_size * 3, 3), dtype=np.uint8)
    base_tile = cv2.resize(cropped_face, (sub_size, sub_size))
    
    styles = HAIRSTYLES_LABELS[predicted_class]
    count = 0
    
    for r in range(3):
        for c in range(3):
            tile = base_tile.copy()
            style_name = styles[count]
            
            # --- RENDER UNIQUE HAIRSTYLE GEOMETRIC FILTER OVERLAYS ---
            if "Bangs" in style_name or "Fringe" in style_name:
                cv2.ellipse(tile, (sub_size//2, sub_size//4), (90, 35), 0, 180, 360, (40, 20, 10), -1)
            elif "Pompadour" in style_name or "High" in style_name or "Quiff" in style_name:
                cv2.ellipse(tile, (sub_size//2, sub_size//5), (70, 50), 0, 180, 360, (20, 20, 20), -1)
            elif "Long" in style_name or "Waves" in style_name:
                cv2.rectangle(tile, (35, sub_size//3), (70, sub_size), (15, 30, 80), -1)
                cv2.rectangle(tile, (sub_size-70, sub_size//3), (sub_size-35, sub_size), (15, 30, 80), -1)
            elif "Bob" in style_name or "Lob" in style_name:
                cv2.ellipse(tile, (sub_size//2, sub_size//3), (110, 60), 0, 180, 360, (30, 60, 15), -1)
                cv2.rectangle(tile, (30, sub_size//3), (60, int(sub_size*0.8)), (30, 60, 15), -1)
                cv2.rectangle(tile, (sub_size-60, sub_size//3), (sub_size-30, int(sub_size*0.8)), (30, 60, 15), -1)
            else:
                cv2.circle(tile, (sub_size//2, sub_size//4), 80, (10, 10, 10), 4)

            # Insert Clean Text Labels on the Grid Cards
            cv2.rectangle(tile, (0, sub_size-35), (sub_size, sub_size), (0, 0, 0), -1)
            cv2.putText(tile, style_name, (15, sub_size-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            grid_img[r*sub_size:(r+1)*sub_size, c*sub_size:(c+1)*sub_size] = tile
            count += 1
            
    # Render final matrix layout output onto screen
    grid_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    st.image(grid_rgb, caption="Your 3x3 Try-On Layout Matrix Grid", use_container_width=True)
