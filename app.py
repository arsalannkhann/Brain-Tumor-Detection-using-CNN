from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io


app = Flask(__name__)

# Global variables
MODEL_PATH = 'model/best_model.keras'
model = None

def load_ml_model():
    global model
    try:
        # Enable GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # Load model with optimization
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
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_prediction(image):
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Convert image to RGB and resize
        img = image.convert('RGB')
        img = img.resize((299, 299), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, batch_size=1)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return class_names[predicted_class], confidence
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

# Load model at startup
load_ml_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Process image
        image = Image.open(file.stream)
        
        # Get prediction
        predicted_class, confidence = get_prediction(image)
        
        if predicted_class is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        return jsonify({
            'tumor_type': predicted_class,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0')