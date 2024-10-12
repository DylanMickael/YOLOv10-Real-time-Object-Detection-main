from ultralytics import YOLO
import cvzone
import cv2
import os
import time
import numpy as np

model = YOLO('best.pt')

# Variables for limiting captures
last_capture_time = 0
capture_interval = 5  # seconds
last_class_detected_name = None
last_image = None  # To store the previous frame
pixel_diff_threshold = 25000  # Adjust this threshold as needed

def save_image(image, class_name):
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'captures/{class_name}_{timestamp}.png'
    
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")

# cv2.VideoCapture(_cameraUrl_) to connect to another camera
cap = cv2.VideoCapture(0)  

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    results = model(image)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = box.cls[0]
            class_detected_number = int(class_detected_number)
            class_detected_name = results[0].names[class_detected_number]

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
            
            # Check if object is 'toothbrush' and limit captures
            current_time = time.time()
            if class_detected_name == 'accident':
                print("Accident detected!")

                # Capture only if enough time has passed and if it's a new detection
                if (current_time - last_capture_time > capture_interval) or (class_detected_name != last_class_detected_name):

                    # Check if there is a last image and compare pixel differences
                    if last_image is not None:
                        diff = cv2.absdiff(last_image, image)
                        non_zero_count = np.count_nonzero(diff)

                        print(f"Pixel difference: {non_zero_count}")
                        
                        # Only capture if pixel difference exceeds the threshold
                        if non_zero_count > pixel_diff_threshold:
                            save_image(image, class_detected_name)
                            last_capture_time = current_time
                            last_class_detected_name = class_detected_name
                    else:
                        # If no previous image exists, save the first one
                        save_image(image, class_detected_name)
                        last_capture_time = current_time
                        last_class_detected_name = class_detected_name

                    # Update the last image
                    last_image = image.copy()

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
