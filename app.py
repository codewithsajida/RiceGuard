import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# ------------------------------
# Google Drive Model Settings
# ------------------------------
MODEL_PATH = "rice_disease_model.h5"
DRIVE_FILE_ID = "1syroRsKo08V2-tqF-Amvu4PZ4zB8Vxgf"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="RiceGuard - AI Disease Detection",
    page_icon="🌾",
    layout="wide",
)

# ------------------------------
# Disease Info Dictionary
# ------------------------------
disease_info = {
    "Bacterial Leaf Blight": {
        "description": "Causes yellowing and wilting of leaves. Can significantly reduce yield.",
        "symptoms": "Water-soaked streaks, yellow lesions, milky dew drops.",
        "treatment": "Use copper-based fungicides and avoid excessive nitrogen fertilizer.",
        "icon": "🦠",
    },
    "Brown Spot": {
        "description": "Fungal disease causing brown circular spots on leaves.",
        "symptoms": "Round to oval brown spots with yellow halo.",
        "treatment": "Improve soil nutrients and treat seeds with fungicides.",
        "icon": "🍂",
    },
    "Leaf Blast": {
        "description": "Diamond-shaped lesions appear on rice leaves.",
        "symptoms": "Spindle-shaped spots with gray centers.",
        "treatment": "Use systemic fungicides and resistant varieties.",
        "icon": "🔥",
    },
    "Leaf Smut": {
        "description": "Black powdery masses on leaves.",
        "symptoms": "Small black linear lesions on leaf blades.",
        "treatment": "Remove infected debris and apply fungicides if severe.",
        "icon": "🌑",
    },
    "Narrow Brown Leaf Spot": {
        "description": "Long narrow brown streaks, usually late season.",
        "symptoms": "Linear brown streaks on leaves.",
        "treatment": "Use resistant varieties and balanced fertilization.",
        "icon": "🌾",
    },
    "Healthy Rice Leaf": {
        "description": "Plant is healthy with no visible disease.",
        "symptoms": "Green, vibrant leaves with no spots.",
        "treatment": "Continue regular monitoring and maintenance.",
        "icon": "✅",
    },
    "Unknown": {
        "description": "Disease not recognized by the system.",
        "symptoms": "Please consult an agricultural expert.",
        "treatment": "Avoid applying random chemicals; seek guidance.",
        "icon": "❓",
    }
}

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found! Disease Detection page may not work.")
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    else:
        class_names = list(disease_info.keys())
    return model, class_names

# ------------------------------
# Preprocess Image
# ------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img) / 255.0
    return np.expand_dims(img_arr, axis=0)

# ------------------------------
# Authentication Pages
# ------------------------------
def signup_page():
    st.title("Create an Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if username.strip() == "" or password.strip() == "":
            st.error("Please fill all fields.")
        elif username in st.session_state.user_data:
            st.error("Username already exists!")
        else:
            st.session_state.user_data[username] = password
            # Save to permanent JSON
            with open("user_data.json", "w") as f:
                json.dump(st.session_state.user_data, f)
            st.success("Signup successful! You can now login.")

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.user_data and st.session_state.user_data[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.experimental_rerun = lambda: None  # Avoid old rerun
            st.stop()  # Stop execution to refresh session
        else:
            st.error("Invalid username or password!")

# ------------------------------
# Main App
# ------------------------------
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_data" not in st.session_state:
        # Load permanent JSON if exists
        if os.path.exists("user_data.json"):
            with open("user_data.json", "r") as f:
                st.session_state.user_data = json.load(f)
        else:
            st.session_state.user_data = {}

    # ---------------- Sidebar Account ----------------
    st.sidebar.title("🌾 RiceGuard Account")
    if not st.session_state.logged_in:
        choice = st.sidebar.radio("Choose", ["Login", "Sign Up"])
        if choice == "Login":
            login_page()
        else:
            signup_page()
        return

    # Logout button
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully. Please refresh the page to login again.")
        st.stop()

    # ---------------- Navigation ----------------
    app_mode = st.sidebar.radio("Navigate", ["Home", "Disease Detection", "Gallery", "About Us"])

    # Load model only when needed
    model, class_names = None, None
    if app_mode == "Disease Detection":
        model, class_names = load_model()

    # ---------------- Home ----------------
    if app_mode == "Home":
        st.markdown("<h1 style='text-align:center; color:#2E7D32;'>🌾 RiceGuard - AI Disease Detection</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:18px;'>AI-powered rice leaf disease detection using deep learning.</p>", unsafe_allow_html=True)

        st.markdown("### ⭐ Features")
        st.write(
            """
            - Upload rice leaf & detect disease  
            - Detailed description, symptoms & treatment  
            - Beautiful disease gallery  
            - Secure login & logout system  
            - Cloud model loading  
            """
        )

    # ---------------- Disease Detection ----------------
    elif app_mode == "Disease Detection":
        st.header("🔍 Detect Rice Leaf Disease")
        upload = st.file_uploader("Upload a rice leaf image", type=["jpg", "png", "jpeg"])
        if upload:
            st.image(upload, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Now"):
                if model is None:
                    st.error("Model not loaded. Cannot perform detection.")
                else:
                    arr = preprocess_image(upload)
                    pred = model.predict(arr)
                    idx = np.argmax(pred[0])
                    label = class_names[idx].strip()
                    confidence = round(100 * np.max(pred[0]), 2)
                    info = disease_info.get(label, disease_info["Unknown"])

                    st.subheader(f"{info.get('icon', '🌱')} {label}")
                    st.write(f"### Confidence: **{confidence}%**")
                    st.write(f"**Description:** {info.get('description')}")
                    st.write(f"**Symptoms:** {info.get('symptoms')}")
                    st.write(f"**Treatment:** {info.get('treatment')}")

                    if "Healthy" in label:
                        st.success("🌱 This rice leaf is healthy!")
                    else:
                        st.error("⚠ Disease detected!")

    # ---------------- Gallery ----------------
    elif app_mode == "Gallery":
        st.header("🌿 Disease Gallery")
        for dis, info in disease_info.items():
            with st.expander(f"{info['icon']} {dis}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Symptoms:** {info['symptoms']}")
                st.write(f"**Treatment:** {info['treatment']}")

    # ---------------- About Us ----------------
    elif app_mode == "About Us":
        st.header("🌾 About RiceGuard")
        st.write("""
        RiceGuard is an AI-powered application designed to help farmers 
        and agricultural researchers detect rice leaf diseases efficiently. 
        Using deep learning models, RiceGuard analyzes uploaded images of rice leaves, 
        providing detailed disease information, symptoms, and suggested treatments.
        """)
        st.subheader("Our Mission")
        st.write("To empower sustainable farming and reduce crop losses through AI technology.")
        st.subheader("Contact Us")
        st.write("Sawaira Iqbal")

    # ---------------- Footer ----------------
    st.markdown(
        "<hr><p style='text-align:center; font-size:12px; color:gray;'>&copy; 2025 RiceGuard | Developed by Your Name</p>",
        unsafe_allow_html=True
    )

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    main()
