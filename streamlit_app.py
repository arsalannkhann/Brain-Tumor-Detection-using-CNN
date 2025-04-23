import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------ Config ------------------ #
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
GDRIVE_FILE_ID = "1aEc1Ni1mds5anu28giaiXkcM9_OOxV2y"  # Replace with your actual file ID
MODEL_PATH = "best_model.keras"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------ Model Downloader ------------------ #
@st.cache_resource
def load_dl_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")

    # GPU memory growth (optional)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=False)

    # Warm-up
    dummy = np.zeros((1, 299, 299, 3))
    model.predict(dummy)

    return model

model = load_dl_model()

# ------------------ Prediction Function ------------------ #
def predict_class(image: Image.Image):
    try:
        img = image.convert("RGB").resize((299, 299), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        class_index = np.argmax(prediction)
        confidence = float(prediction[class_index])
        return CLASS_NAMES[class_index], confidence
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None, None

# ------------------ Sample Image Display & Prediction ------------------ #
def show_sample_and_predict():
    st.subheader("🧪 Try with a Sample Image")

    sample_dir = "samples"
    sample_options = []

    for class_name in CLASS_NAMES:
        image_paths = glob(os.path.join(sample_dir, class_name, "*"))
        for img_path in image_paths:
            label = f"{class_name} - {os.path.basename(img_path)}"
            sample_options.append((label, img_path))

    selected_label = st.selectbox("Choose a sample image:", [opt[0] for opt in sample_options])
    selected_image_path = dict(sample_options)[selected_label]

    if selected_image_path:
        image = Image.open(selected_image_path)
        st.image(image, caption="Selected Sample", use_container_width=True)

        if st.button("🔍 Predict Sample"):
            predicted_class, confidence = predict_class(image)
            st.success(f"🎯 Tumor Type: {predicted_class.upper()}")
            st.info(f"Confidence: {confidence:.2f}")

# ------------------ Uploader UI ------------------ #
def uploader_section():
    st.subheader("📤 Upload an MRI Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Uploaded Image"):
            predicted_class, confidence = predict_class(image)
            st.success(f"🎯 Tumor Type: {predicted_class.upper()}")
            st.info(f"Confidence: {confidence:.2f}")

# ------------------ Streamlit App Layout ------------------ #
def main():
    
    st.title("🧠 Brain Tumor Classification App")
    st.markdown("Predicts tumor type: `Glioma`, `Meningioma`, `No Tumor`, or `Pituitary` using a deep learning model.")

    uploader_section()
    st.markdown("---")
    show_sample_and_predict()

if __name__ == "__main__":
    main()
