from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set environment variable to use Python implementation of protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Global variable for the TensorFlow Lite interpreter
interpreter = None

# Function to load the model
def load_model():
    global interpreter
    try:
        # Use absolute path or ensure the model is in the correct directory
        model_path = os.path.join(os.path.dirname(__file__), 'model.tflite')
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Home route
@app.route('/')
def home():
    return "Waste Classification AI Server is running!"

# Classify route
@app.route('/classify', methods=['POST'])
def classify_image():
    global interpreter
    
    # Check if interpreter is loaded
    if interpreter is None:
        interpreter = load_model()
        if interpreter is None:
            return jsonify({'error': 'Failed to load model'}), 500
    
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
        
        # Normalize the image (convert to float and scale)
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
        
        # Predefined class names (adjust based on your model)
        class_names = ["Plastic", "Paper", "Metal", "WetWaste"]
        
        # Get the index of the highest confidence
        prediction_index = np.argmax(results)
        confidence = float(results[prediction_index]) / 255  # Convert to percentage
        
        # Get the predicted class name
        predicted_class = class_names[prediction_index]

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_scores': results.tolist()
        })
        
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500

# Ensure model is loaded at startup
interpreter = load_model()

# Corrected main block
if __name__ == '__main__':
    # Load model before starting the server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
