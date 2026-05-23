import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

st.set_page_config(page_title="Real-Time AI Hairstyle Matrix", page_icon="✂️", layout="wide")

# 1. Pull Trained Weights Securely
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
# Pre-draw clean mockup geometries using OpenCV for our 9 matrix elements
HAIRSTYLES_LABELS = {
    'Heart':  ["Side Fringe", "Long Layers", "Pixie Cut", "Wispy Bangs", "Bob Cut", "Taper Fade", "Textured Crop", "Curtain Bangs", "Classic Pompadour"],
    'Oblong': ["Side Part", "Blunt Bangs", "Voluminous Curls", "Shag Cut", "Chin Bob", "Wavy Lob", "Slick Back", "Buzz Cut", "Modern Quiff"],
    'Oval':   ["Blunt Bob", "Slick Undercut", "Long Waves", "French Crop", "Crew Cut", "Layered Shag", "Curtain Hair", "Side Swept", "Fringe Crop"],
    'Round':  ["High Pompadour", "Asymmetric Bob", "Long Shag", "Spiky Texture", "Undercut Fade", "Side Bangs", "Faux Hawk", "Textured Quiff", "Ivy League"],
    'Square': ["Layered Waves", "Side Swept Bob", "Clean Buzz", "Crew Cut", "Textured Crop", "Classic Part", "Comb Over", "Long Shag", "Slicked Back"]
}

# 3. Real-Time Video Grid Pipeline Engine
class HairstyleGridProcessor:
    def __init__(self):
        self.frame_count = 0
        self.current_face_shape = "Oval" # Default fall-back target

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape
        
        # Crop to square aspect ratio for balanced grid dimensions
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped_face = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Optimization: Classify every 15 frames to maintain smooth framerates
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            try:
                resized = cv2.resize(cropped_face, (224, 224))
                normalized = resized / 255.0
                tensor_input = np.expand_dims(normalized, axis=0)
                preds = model.predict(tensor_input, verbose=0)
                self.current_face_shape = CLASSES[np.argmax(preds)]
            except:
                pass

        # Generate the live 3x3 filter grid layout matrix
        sub_size = 250
        grid_img = np.zeros((sub_size * 3, sub_size * 3, 3), dtype=np.uint8)
        base_tile = cv2.resize(cropped_face, (sub_size, sub_size))
        
        styles = HAIRSTYLES_LABELS[self.current_face_shape]
        
        count = 0
        for r in range(3):
            for c_idx in range(3):
                tile = base_tile.copy()
                style_name = styles[count]
                
                # --- FILTER OVERLAY GEOMETRY INJECTIONS ---
                # To simulate hair filters without external face meshes, we apply 
                # custom graphic bounding contours based on style properties
                if "Bangs" in style_name or "Fringe" in style_name:
                    cv2.ellipse(tile, (sub_size//2, sub_size//4), (80, 30), 0, 180, 360, (40, 20, 10), -1)
                elif "Pompadour" in style_name or "High" in style_name or "Quiff" in style_name:
                    cv2.ellipse(tile, (sub_size//2, sub_size//5), (60, 45), 0, 180, 360, (20, 20, 20), -1)
                elif "Long" in style_name or "Waves" in style_name:
                    cv2.rectangle(tile, (30, sub_size//3), (60, sub_size), (15, 30, 80), -1)
                    cv2.rectangle(tile, (sub_size-60, sub_size//3), (sub_size-30, sub_size), (15, 30, 80), -1)
                elif "Bob" in style_name or "Lob" in style_name:
                    cv2.ellipse(tile, (sub_size//2, sub_size//3), (95, 50), 0, 180, 360, (30, 60, 15), -1)
                    cv2.rectangle(tile, (25, sub_size//3), (50, int(sub_size*0.75)), (30, 60, 15), -1)
                    cv2.rectangle(tile, (sub_size-50, sub_size//3), (sub_size-25, int(sub_size*0.75)), (30, 60, 15), -1)
                else: # Generic texturing crown contour for short styles
                    cv2.circle(tile, (sub_size//2, sub_size//4), 70, (10, 10, 10), 3)

                # Render Text Labels cleanly into the frame matrix panel
                cv2.rectangle(tile, (0, sub_size-30), (sub_size, sub_size), (0, 0, 0), -1)
                cv2.putText(tile, style_name, (10, sub_size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Position tile into the core final matrix grid alignment
                grid_img[r*sub_size:(r+1)*sub_size, c_idx*sub_size:(c_idx+1)*sub_size] = tile
                count += 1
                
        # Draw Master Header detailing global face shape tracking parameters
        output_frame = cv2.copyMakeBorder(grid_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 24, 28))
        cv2.putText(output_frame, f"LIVE TRACKING DETECTED: {self.current_face_shape.upper()} FACE STRUCTURE", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 110), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

# 4. User Interface Architecture
st.title("Live Real-Time Hairstyle Filter Matrix ⚡")
st.write("Activate your camera below. The AI will classify your facial structure structure real-time and render a 3x3 live filter overlay grid.")

# Setup ICE configurations to ensure streaming stability on public clouds
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="hairstyle-matrix-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=HairstyleGridProcessor,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False}
)
