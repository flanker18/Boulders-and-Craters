from ultralytics import YOLO
import cv2
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import traceback

app = Flask(__name__)
CORS(app)

# Load the model globally
model = YOLO(r"D:\Projects\Boulders_Craters\craters - boulders.v1i.yolov8\best.pt")

def process_image(image_array, conf_threshold=0.25):
    """
    Process a single image array and return the detection results
    """
    try:
        # Ensure image is in RGB format
        if isinstance(image_array, Image.Image):
            image_array = np.array(image_array)
        
        # Run prediction
        results = model.predict(
            source=image_array,
            conf=conf_threshold,
            save=False
        )[0]
        
        # Convert the image to BGR for OpenCV operations
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence score
            label = f'{class_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(image, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Convert back to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read the image file
        image_file = request.files['image']
        
        # Print debug information
        print(f"Received image: {image_file.filename}")
        
        # Read and convert image
        image = Image.open(image_file).convert('RGB')
        
        # Process the image
        processed_image = process_image(image)
        
        # Convert the processed image to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_str}'
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)