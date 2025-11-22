import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="RiceGuard - AI Disease Detection", page_icon="🌾", layout="wide")

# --- Disease Info ---
disease_info = {
    "Bacterial Leaf Blight": {"description": "Causes yellowing and wilting of leaves. It can significantly reduce yield.",
                               "symptoms": "Water-soaked streaks, yellow lesions, milky dew drops.",
                               "treatment": "Use copper-based fungicides and avoid excess nitrogen fertilizer.",
                               "icon": "🦠"},
    "Brown Spot": {"description": "A fungal disease that causes brown circular spots on leaves.",
                   "symptoms": "Round to oval brown spots with a yellow halo.",
                   "treatment": "Improve soil nutrients and treat seeds with fungicides.",
                   "icon": "🍂"},
    "Leaf Blast": {"description": "Causes diamond-shaped lesions.",
                   "symptoms": "Spindle-shaped spots with gray centers.",
                   "treatment": "Plant resistant varieties and use systemic fungicides.",
                   "icon": "🔥"},
    "Leaf Smut": {"description": "Creates black powdery masses on leaves.",
                  "symptoms": "Small, black, linear lesions on leaf blades.",
                  "treatment": "Remove infected plant debris and apply fungicides if severe.",
                  "icon": "🌑"},
    "Narrow Brown Leaf Spot": {"description": "Linear brown lesions on leaves, often late in the season.",
                               "symptoms": "Short, linear, narrow brown streaks.",
                               "treatment": "Use resistant varieties and balanced fertilization.",
                               "icon": "🌾"},
    "Healthy Rice Leaf": {"description": "Plant is in good condition with no visible signs of disease.",
                          "symptoms": "Green, vibrant leaves with no spots.",
                          "treatment": "Continue regular maintenance and monitoring.",
                          "icon": "✅"}
}

# --- Load Model ---
@st.cache_resource
def load_system():
    if not os.path.exists('rice_disease_model.h5'):
        st.warning("Model not found! Please train the model first.")
        return None, None
    model = tf.keras.models.load_model('rice_disease_model.h5')
    if os.path.exists('class_names.json'):
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    else:
        class_names = list(disease_info.keys())
    return model, class_names

# --- Main App ---
def main():
    st.sidebar.title("🌾 Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Disease Detection", "Gallery & Info"])

    model, class_names = load_system()
    if model is None:
        return  # Stop execution if model is missing

    if app_mode == "Home":
        st.markdown("<h1 style='text-align:center;'>Rice Plant Disease Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Upload a rice leaf image to detect diseases.</p>", unsafe_allow_html=True)

    elif app_mode == "Disease Detection":
        st.markdown("## 🔎 Detect Disease")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Open image for display
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Leaf", use_container_width=True)

            if st.button("Analyze Now"):
                # Reset file pointer for OpenCV
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if opencv_image is None:
                    st.error("Error reading image! Please upload a valid image file.")
                    return

                # Preprocess for model
                opencv_image = cv2.resize(opencv_image, (224, 224))
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                img_array = np.expand_dims(opencv_image / 255.0, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                predicted_label = class_names[predicted_idx].strip()  # remove extra spaces
                confidence = round(100 * np.max(predictions[0]), 2)

                # Match label to disease info
                predicted_label_clean = predicted_label.title()  # ensures proper capitalization
                info = disease_info.get(predicted_label_clean, {})
                icon = info.get("icon", "🌱")
                desc = info.get("description", "No description available.")
                treat = info.get("treatment", "Consult an expert.")

                if "Healthy" in predicted_label_clean:
                    st.success(f"{icon} {predicted_label_clean} (Confidence: {confidence}%)\n{desc}")
                else:
                    st.error(f"{icon} {predicted_label_clean} Detected (Confidence: {confidence}%)\n{desc}\nTreatment: {treat}")

    elif app_mode == "Gallery & Info":
        st.markdown("## 🌿 Disease Gallery")
        for disease, info in disease_info.items():
            with st.expander(f"{info['icon']} {disease}"):
                st.write(f"**About:** {info['description']}")
                st.write(f"**Symptoms:** {info['symptoms']}")
                st.write(f"**Treatment:** {info['treatment']}")

if __name__ == "__main__":
    main()
