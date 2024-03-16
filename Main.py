import pickle
from skimage.transform import resize
import numpy as np
import cv2

model = pickle.load(open('model.p', 'rb'))
cap = cv2.VideoCapture('Data/Video/parking_1920_1080_loop.mp4')

mask_image = cv2.imread('Data//Video//mask.png', cv2.IMREAD_GRAYSCALE)
_, thresholded_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Generate parking spot map
parking_spot_map = {}
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    parking_spot_map[f"spot_{i+1}"] = (x, y, x+w, y+h)

# Function to determine if parking spot is empty or not
def empty_or_not(frame, spot_coords):
    x1, y1, x2, y2 = spot_coords
    spot_roi = frame[y1:y2, x1:x2]
    img_resized = resize(spot_roi, (15, 15, 3))
    flat_data = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(flat_data)
    return prediction == 0

skip_patterns = [1, 5, 10, 20, 30]

frame_count = 0
pattern_index = 0

prev_spot_occupancy = {spot_name: None for spot_name in parking_spot_map.keys()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_spots = len(parking_spot_map)
    available_spots = total_spots

    if frame_count % skip_patterns[pattern_index] != 0:
        frame_count += 1
        continue

    pattern_index = (pattern_index + 1) % len(skip_patterns)
    frame_count += 1
   
    # Process each parking spot in the parking spot map
    for spot_name, spot_coords in parking_spot_map.items():
        is_empty = empty_or_not(frame, spot_coords)
        color = (0, 255, 0) if is_empty else (0, 0, 255)
        x1, y1, x2, y2 = spot_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not is_empty:
            available_spots -= 1

        if prev_spot_occupancy[spot_name] != is_empty:
            if is_empty:
                available_spots += 1
            else:
                available_spots -= 1 

        prev_spot_occupancy[spot_name] = is_empty
    
    text = f'Available Spots: {available_spots}/{total_spots}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with parking spots highlighted
    cv2.imshow('Parking Space Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


