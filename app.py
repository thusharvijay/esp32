from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Set environment variable to use Python implementation of protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Global variable for the TensorFlow Lite interpreter
interpreter = None

# Function to load the model
def load_model():
    global interpreter
    try:
        model_path = 'model.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a dummy interpreter for testing if model file is missing
        if not os.path.exists(model_path):
            print("Model file not found. Using dummy model for testing.")

# Home route
@app.route('/')
def home():
    return "ESP32-CAM Image Classification Server is running!"

# Classify route
@app.route('/classify', methods=['POST'])
def classify_image():
    global interpreter
    
    # Check if interpreter is loaded
    if interpreter is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
    
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Get the base64 encoded image
        img_data = request.json['image']
        
        # Decode the base64 image
        img_bytes = base64.b64decode(img_data)
        
        # Open the image with PIL
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize the image to match model input
        img = img.resize((224, 224))
        
        # Convert to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Normalize the image
        img_array = np.array(img, dtype=np.uint8)
        
        # Reshape for the model input
        input_tensor = np.expand_dims(img_array, axis=0)
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the prediction results
        results = np.squeeze(output_data)
        
        # For this example, let's say we have categories like "garbage", "recycling", "compost"
        # Adjust these class names based on your actual model
        class_names = ["Plastic", "Paper", "Metal", "WetWaste"]
        
        # Get the index of the highest confidence
        prediction_index = np.argmax(results)
        confidence = float(results[prediction_index])
        
        # Get the predicted class name
        predicted_class = class_names[prediction_index]
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_scores': results.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Try to load the model at startup
load_model()

if __name__ == '__main__':
    # Load model before starting the server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
