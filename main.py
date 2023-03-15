from flask import Flask, jsonify, request
import torch
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics-yolov5-v7.0-120-g3e55763/yolov5', 'custom', path='best.pt', force_reload=True)

# Define object detection endpoint
@app.route('/detect', methods=['POST'])
def detect():
    # Get image from request
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    # Run object detection on image
    results = model(image)

    # Convert results to JSON
    output = {'objects': []}
    for obj in results.xyxy[0]:
        output['objects'].append({
            'label': obj[5],
            'confidence': obj[4],
            'bbox': obj[:4].tolist()
        })

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True,port=4000)
