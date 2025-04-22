import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Google Drive file ID of your Keras model
GDRIVE_FILE_ID = "1aEc1Ni1mds5anu28giaiXkcM9_OOxV2y"
MODEL_PATH = "best_model.keras"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to download model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Cached function to load the model once
@st.cache_resource
def load_brain_model():
    download_model()

    # Allow GPU memory growth if GPU available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    model = load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )

    # Warm up the model
    dummy_input = np.zeros((1, 299, 299, 3))
    model.predict(dummy_input)

    return model

# Prediction function
def get_prediction(image, model):
    try:
        # Convert image to RGB and resize
        img = image.convert('RGB')
        img = img.resize((299, 299), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, batch_size=1)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        return CLASS_NAMES[predicted_class], confidence
    except Exception as e:
        return None, str(e)

# Streamlit UI
def main():
    st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
    st.title("🧠 Brain Tumor Detection using Deep Learning")
    st.markdown("Upload an MRI scan image to predict the tumor type.")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading model and predicting..."):
            model = load_brain_model()
            predicted_class, confidence = get_prediction(image, model)

        if predicted_class:
            st.success(f"🎯 Predicted Tumor Type: **{predicted_class.upper()}**")
            st.info(f"Confidence: `{confidence * 100:.2f}%`")
        else:
            st.error("❌ Could not make a prediction.")

if __name__ == "__main__":
    main()
