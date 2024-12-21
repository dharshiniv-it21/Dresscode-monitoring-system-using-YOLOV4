from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import signal
import numpy as np
import sys

app = Flask(__name__)

# Load the YOLOv model
try:
    model = YOLO("best.pt")  # Update with your model path
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

def detect_objects(frame):
    # Perform object detection on the frame
    results = model.predict(frame)

    # Draw the detection results on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results[0].names[int(box.cls[0])]
        confidence = round(box.conf[0].item(), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def gen_frames():
    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Perform object detection
        frame = detect_objects(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            break
        frame = buffer.tobytes()

        # Yield the frame in the format required for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('contact.html')  # Ensure this points to your HTML file

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

class Detection:
    def analyze_dress_code(self, image):
        # Placeholder logic for dress code analysis
        detected_items = self.detect_items(image)
        proper_dress_code = self.check_dress_code(detected_items)
        return {'status': 'Proper Dresscode' if proper_dress_code else 'Improper DressCode'}

    def detect_items(self, image):
        return ['proper', 'improper']  # Placeholder

    def check_dress_code(self, items):
        return 'proper' in items and 'improper' in items

detection = Detection()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    # Read the image file
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    
    # Analyze the image for dress code
    result = detection.analyze_dress_code(img)
    
    return jsonify(result)

def cleanup(signum, frame):
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Clean up the camera when the application stops
cap.release()