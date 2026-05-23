import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
import os
import zipfile
import mediapipe as mp
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="Barbershop AI Style Matrix", page_icon="✂️", layout="wide")

# 1. Fetch AI Weights & High-Fidelity Barbershop Hair Assets
@st.cache_resource
def load_project_resources():
    MODEL_PATH = 'hairstyle_model.h5'
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Core AI classification model... Please wait."):
            MODEL_ID = "1BrcUSYWTHb4F2Oz3h_wapVqiQadXZhpC" 
            URL = f"https://google.com{MODEL_ID}"
            urllib.request.urlretrieve(URL, MODEL_PATH)
            
    ASSETS_DIR = 'hairstyles'
    ZIP_PATH = 'hairstyles.zip'
    if not os.path.exists(ASSETS_DIR):
        with st.spinner("Downloading high-fidelity realistic hair asset library..."):
            # New Zip Package containing pre-aligned real human hairstyle textures
            ZIP_ID = "1gO_Z9NskkYwD7w6C_Y9kLx89SgQXp3H1" 
            ZIP_URL = f"https://google.com{ZIP_ID}"
            urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall('.')
                
    return tf.keras.models.load_model(MODEL_PATH)

model = load_project_resources()
CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

HAIRSTYLES_LABELS = {
    'Heart':  ["Side Fringe", "Long Layers", "Pixie Cut", "Wispy Bangs", "Bob Cut", "Taper Fade", "Textured Crop", "Curtain Bangs", "Classic Pompadour"],
    'Oblong': ["Side Part", "Blunt Bangs", "Voluminous Curls", "Shag Cut", "Chin Bob", "Wavy Lob", "Slick Back", "Buzz Cut", "Modern Quiff"],
    'Oval':   ["Blunt Bob", "Slick Undercut", "Long Waves", "French Crop", "Crew Cut", "Layered Shag", "Curtain Hair", "Side Swept", "Fringe Crop"],
    'Round':  ["High Pompadour", "Asymmetric Bob", "Long Shag", "Spiky Texture", "Undercut Fade", "Side Bangs", "Faux Hawk", "Textured Quiff", "Ivy League"],
    'Square': ["Layered Waves", "Side Swept Bob", "Clean Buzz", "Crew Cut", "Textured Crop", "Classic Part", "Comb Over", "Long Shag", "Slicked Back"]
}

# 3. Barbershop-Style Blending Engine
def apply_hair_filter(face_img, hair_path, landmark_points):
    h, w, _ = face_img.shape
    
    # MediaPipe Forehead anchor nodes
    top_head = landmark_points[10]       
    left_temple = landmark_points[234]   
    right_temple = landmark_points[454]  
    
    # Calculate scale dynamically
    hair_width = int(np.linalg.norm(np.array([left_temple[0]*w, left_temple[1]*h]) - np.array([right_temple[0]*w, right_temple[1]*h])) * 1.75)
    if hair_width <= 0: hair_width = 100
        
    hair_overlay = cv2.imread(hair_path, cv2.IMREAD_UNCHANGED)
    if hair_overlay is None:
        return face_img.copy()
        
    oh, ow, _ = hair_overlay.shape
    hair_height = int(hair_width * (oh / ow))
    hair_resized = cv2.resize(hair_overlay, (hair_width, hair_height), interpolation=cv2.INTER_LANCZOS4)
    
    pos_x = int(top_head[0] * w) - (hair_width // 2)
    pos_y = int(top_head[1] * h) - (hair_height // 2) + int(h * 0.03) # Lowered anchor to sit naturally on hairline
    
    output_tile = face_img.copy()
    
    # BARBERSHOP INTERPOLATION COLOR MATCHING
    # Sample background face lighting conditions to color-correct hair strands
    face_sample = face_img[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)]
    if face_sample.size > 0:
        mean_face = cv2.mean(face_sample)[:3]
        # Softly tint the hair channels to blend with environmental illumination
        hair_resized[:, :, 0] = np.clip(hair_resized[:, :, 0] * (mean_face[0]/110.0), 0, 255)
        hair_resized[:, :, 1] = np.clip(hair_resized[:, :, 1] * (mean_face[1]/110.0), 0, 255)
        hair_resized[:, :, 2] = np.clip(hair_resized[:, :, 2] * (mean_face[2]/110.0), 0, 255)

    # Feathered Alpha Channel Matrix Blending
    for r in range(hair_height):
        for c in range(hair_width):
            if pos_y + r >= h or pos_x + c >= w or pos_y + r < 0 or pos_x + c < 0:
                continue
                
            # Read image alpha channel spectrum
            alpha_raw = hair_resized[r, c, 3]
            
            # Applying edge softening gradient
            if alpha_raw > 0:
                alpha = (alpha_raw / 255.0) * 0.95 # Soft ceiling to ensure skin pores show through fringes
                output_tile[pos_y + r, pos_x + c] = (1.0 - alpha) * output_tile[pos_y + r, pos_x + c] + alpha * hair_resized[r, c, :3]
                
    return output_tile

# 4. Interface Controls Setup
st.title("Barbershop AI Hairstyle Transformation Matrix 💇‍♂️")
st.write("Take a snapshot. MediaPipe maps your face metrics to seamlessly morph high-fidelity hair models directly into your hairline!")

cam_image = st.camera_input("Center your head inside the view module camera box")

if cam_image is not None:
    image = Image.open(cam_image)
    img_cv = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Exact center square crop factor calculation logic
    h, w, _ = img_cv.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_face = img_cv[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    rgb_crop = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb_crop)
    
    if not mesh_results.multi_face_landmarks:
        st.error("❌ MediaPipe could not resolve your facial proportions. Face forward under bright lighting.")
        st.stop()
        
    face_landmarks = mesh_results.multi_face_landmarks.landmark
    landmark_list = [(lm.x, lm.y) for lm in face_landmarks]
    
    with st.spinner("AI is calculating structure contour points..."):
        resized = cv2.resize(cropped_face, (224, 224))
        normalized = resized / 255.0
        tensor_input = np.expand_dims(normalized, axis=0)
        preds = model.predict(tensor_input, verbose=0)
        predicted_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.success(f"### Target Match Identified: **{predicted_class}** ({confidence:.1f}% Confidence Profile)")
    st.write("### Your Custom 3x3 Style Try-On Portfolio:")
    
    # 5. Build Final 3x3 Composite Output Grid
    sub_size = 350 # Expanded resolution sizing for crisp image profiles
    grid_img = np.zeros((sub_size * 3, sub_size * 3, 3), dtype=np.uint8)
    base_tile = cv2.resize(cropped_face, (sub_size, sub_size))
    
    remeshed = face_mesh.process(cv2.cvtColor(base_tile, cv2.COLOR_BGR2RGB))
    tile_landmarks = [(lm.x, lm.y) for lm in remeshed.multi_face_landmarks.landmark]
    
    styles_list = HAIRSTYLES_LABELS[predicted_class]
    count = 0
    
    for r in range(3):
        for c in range(3):
            style_idx = count + 1
            hair_file_path = f"hairstyles/style{style_idx}.png"
            style_name = styles_list[count]
            
            # Render realistic try-on texture overlay
            tile_output = apply_hair_filter(base_tile, hair_file_path, tile_landmarks)
            
            # Format bottom card label styling
            cv2.rectangle(tile_output, (0, sub_size-35), (sub_size, sub_size), (0, 0, 0), -1)
            cv2.putText(tile_output, style_name, (15, sub_size-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            grid_img[r*sub_size:(r+1)*sub_size, c*sub_size:(c+1)*sub_size] = tile_output
            count += 1
            
    grid_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    st.image(grid_rgb, caption="Your 3x3 Barbershop Try-On Layout Portfolio", use_container_width=True)
