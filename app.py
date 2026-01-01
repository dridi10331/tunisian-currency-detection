from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

# Map class IDs to currency values (same as yolo11.py)
CLASS_TO_VALUE = {
    0: 10,   # class_0 -> 10 DT (confirmed: blue bill)
    1: 20,   # class_1 -> 20 DT (confirmed)
    2: 5,    # class_2 -> 5 DT
    3: 50,   # class_3 -> 50 DT
}

CLASS_TO_NAME = {
    0: '10 DT',
    1: '20 DT',
    2: '5 DT',
    3: '50 DT',
}

# Global variables for webcam
camera = None
current_total = 0
detected_bills = []

def process_frame(frame):
    """Process a frame and return annotated frame with total"""
    global current_total, detected_bills
    
    results = model(frame, conf=0.5)
    
    total = 0
    bills = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            
            value = CLASS_TO_VALUE.get(class_id, 0)
            name = CLASS_TO_NAME.get(class_id, f'class_{class_id}')
            
            total += value
            bills.append({
                'denomination': name,
                'value': value,
                'confidence': round(confidence * 100, 1)
            })
    
    current_total = total
    detected_bills = bills
    
    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Add total overlay
    cv2.rectangle(annotated_frame, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Total: {total} DT", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    return annotated_frame

def generate_frames():
    """Generate frames for video streaming"""
    global camera
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        annotated_frame = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_total')
def get_total():
    """Get current total and detected bills"""
    return jsonify({
        'total': current_total,
        'bills': detected_bills,
        'count': len(detected_bills)
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Process uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read image
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process
    results = model(image, conf=0.5)
    
    total = 0
    bills = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            
            value = CLASS_TO_VALUE.get(class_id, 0)
            name = CLASS_TO_NAME.get(class_id, f'class_{class_id}')
            
            total += value
            bills.append({
                'denomination': name,
                'value': value,
                'confidence': round(confidence * 100, 1)
            })
    
    # Annotate and encode
    annotated = results[0].plot()
    cv2.rectangle(annotated, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.putText(annotated, f"Total: {total} DT", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'total': total,
        'bills': bills,
        'count': len(bills),
        'image': f'data:image/jpeg;base64,{img_base64}'
    })

@app.route('/stop_camera')
def stop_camera():
    """Stop webcam"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'Camera stopped'})

if __name__ == '__main__':
    # Create templates folder if not exists
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
