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

# --- NATIVE LOCKING LAYOUT MATRIX LOGIC ---
# Initialize session state tracking parameters
if "gender_selected" not in st.session_state:
    st.session_state.gender_selected = False
if "gender" not in st.session_state:
    st.session_state.gender = "men"

# If the category hasn't been set, intercept the render stack with a stylized selection card
if not st.session_state.gender_selected:
    st.write("---")
    with st.container():
        st.markdown("### 👋 Welcome to HAIR WE GO!")
        st.info("Please complete the setup step below to unlock the camera scanner and hairstyle filters.")
        
        # Display selection tool
        category_choice = st.selectbox(
            "Select Your Gender Category:",
            ["-- Choose Options --", "Men's Hairstyles", "Women's Hairstyles"]
        )
        
        # Determine if the button should be disabled (True if placeholder is selected)
        is_disabled = (category_choice == "-- Choose Options --")
        
        # The button dynamically locks/unlocks based on the 'is_disabled' variable status
        if st.button("Confirm", type="primary", disabled=is_disabled):
            # Map choice directly to folder structures
            st.session_state.gender = "men" if category_choice == "Men's Hairstyles" else "women"
            st.session_state.gender_selected = True
            st.rerun()
                
    st.stop() # Stops execution here so nothing else shows up until selection is made
    
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
    st.write(f"### 📋 Step 3: Your Suggested {st.session_state.gender.capitalize()} Hairstyles")

    
    # Format strings strictly to strip out invisible break flags
    shape_folder = str(detected_shape).lower().strip()
    gender_file = str(st.session_state.gender).lower().strip()

    
    # List all common extensions to check sequentially
    possible_extensions = ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']
    recommendation_path = None
    
    for ext in possible_extensions:
        test_path = os.path.join(ASSET_DIR, f"{shape_folder}/{gender_file}{ext}")
        if os.path.exists(test_path):
            recommendation_path = test_path
            break  # Stop checking once we find a match!

    # Render the discovered file asset safely to the interface layout
    if recommendation_path is not None:
        recommendation_graphic = Image.open(recommendation_path)
        st.image(recommendation_graphic, use_container_width=True, caption=f"Best styles for {detected_shape.upper()} faces")
        
        # --- FIXED STYLING TIPS MATRIX ---
        st.write("---")
        st.write("### ⚠️ Hairstyles & Haircuts to Avoid")
        
        # Base rules applicable to all genders
        avoidance_tips = {
            'oval': [
                "**Avoid heavy, long straight blunt bangs** that cut straight across your face, as they block your features and make a naturally balanced oval head shape look shorter.",
                "**Avoid hairstyles that add excessive height or volume** directly on the top without any width, which can stretch your facial features and make your face appear artificially long."
            ],
            'round': [
                "**Avoid sleek, flat surfaces** that hug your face line, as they act like a border highlighting the roundness of your cheeks.",
                "**Avoid slicked-back styles or middle parts with zero top volume**, which compress your forehead proportions and make the face shape look even wider."
            ],
            'heart': [
                "**Avoid heavy top volume or high pompadours**, as they add bulk to an already wide forehead line and accent a narrow chin.",
                "**Avoid short, blunt-cut wispy bangs** or styles that end harshly right at your cheekbone levels, which can visually expand the upper half of your face."
            ],
            'square': [
                "**Avoid sharp, blunt-cut straight fringes or geometric box layouts**, as these parallel lines emphasize harsh jawline edges and make your face look boxy.",
                "**Avoid slicked-back look variations or center parts with completely flat sides** that offer zero soft volume around the sides of your face."
            ]
        }
        
        # Gender-specific list mapping dictionary
        specific_cuts_to_avoid = {
            'oval': {
                'men': "**❌ SPECIFIC HAIRCUTS TO AVOID:** High and tight mohawks, extremely high pomp-fades with bald-shaved sides, or flat micro-fringes.",
                'women': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Severe slicked-back high ponytails with zero face-framing layers, or bone-straight micro-bangs."
            },
            'round': {
                'men': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Flat, slicked-back side-parts, buzz cuts with zero top texture, or heavy bowl cuts.",
                'women': "**❌ SPECIFIC HAIRCUTS TO AVOID:** The classic blunt chin-length bob, flat pixie cuts with zero top texture, or sleek pageboy styles."
            },
            'heart': {
                'men': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Slicked-back undercut fades, wide straight-across block bangs, or high top-knots.",
                'women': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Short geometric pixie cuts with severe bangs, or harsh chin-length blunt cuts."
            },
            'square': {
                'men': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Geometric flat tops, severe slicked-back buzz cuts with hard lines, or wide square block-fades.",
                'women': "**❌ SPECIFIC HAIRCUTS TO AVOID:** Sharp blunt flapper bobs, severe slicked-back buns, or straight-across heavy blunt fringes."
            }
        }
        
        # Display the formatted tips inside a clean container box
        if shape_folder in avoidance_tips:
            with st.container(border=True):
                st.markdown(f"#### 🚫 Styling Red Flags for {detected_shape.upper()} Profiles ({gender_file.upper()}):")
                
                # Print the general rules
                for tip in avoidance_tips[shape_folder]:
                    st.write(f"- {tip}")
                
                # Fetch gender specific rule text block safely without variable collisions
                if shape_folder in specific_cuts_to_avoid:
                    gender_data = specific_cuts_to_avoid[shape_folder]
                    if gender_file in gender_data:
                        st.write(f"- {gender_data[gender_file]}")
                    else:
                        st.write(f"Debug: gender key '{gender_file}' not found.")
                        
    else:
        st.error(f"❌ Asset file missing inside folder structure! Searched for '{gender_file}' variations inside 'hairstyle_dataset/{shape_folder}/'")
