from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = None

def load_model():
    global interpreter
    model_path = 'model.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Model loaded successfully")

# Load the model when the application starts
load_model()  # Load model on startup instead of using before_first_request

@app.route('/')
def home():
    return "ESP32-CAM Image Classification Server is running!"

@app.route('/classify', methods=['POST'])
def classify_image():
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
        img_array = np.array(img, dtype=np.float32) / 255.0
        
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
        class_names = ["garbage", "recycling", "compost", "other"]
        
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
