from flask import Flask, render_template, Response
import cv2
import pickle
from skimage.transform import resize
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.p', 'rb'))
cap = cv2.VideoCapture('Data/Video/parking_1920_1080_loop.mp4')
if not cap.isOpened:
    print("Error, could not read video capture")
    exit()

mask_image = cv2.imread('Mask/mask.png', cv2.IMREAD_GRAYSCALE)
if mask_image is None:
    print("Error, could not read mask image")
    exit()

_, thresholded_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

parking_spot_map = {}
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    parking_spot_map[f"spot_{i+1}"] = (x, y, x+w, y+h)

def empty_or_not(frame, spot_coords):
    x1, y1, x2, y2 = spot_coords
    spot_roi = frame[y1:y2, x1:x2]
    img_resized = resize(spot_roi, (15, 15, 3))
    flat_data = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(flat_data)
    return prediction == 0

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_spots = len(parking_spot_map)
        available_spots = total_spots

        for spot_name, spot_coords in parking_spot_map.items():
            is_empty = empty_or_not(frame, spot_coords)
            color = (0, 255, 0) if is_empty else (0, 0, 255)
            x1, y1, x2, y2 = spot_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if not is_empty:
                available_spots -= 1
        
        text = f'Available Spots: {available_spots}/{total_spots}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode the frame and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

