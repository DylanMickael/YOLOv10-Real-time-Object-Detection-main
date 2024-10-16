from ultralytics import YOLO
import cvzone
import cv2
import os
import time
import numpy as np

# Load the YOLOv10 model
modelYOLO = YOLO('models/yolov10n.pt')

# Get vocabulary from the model
vocab2 = modelYOLO.names
vocab2_list = list(vocab2.values())

def process_image(image):
    # Use only YOLOv10 for detection
    results_yolo = modelYOLO(image)
    
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab2[class_detected_number]

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

            if class_detected_name in ['car']:
                print("Car detected!")

    return image

image_path = 'Test/images/collision.jfif'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    processed_image = process_image(image)

    # Display the processed image in a window
    cv2.imshow('Processed Image', processed_image)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)  # 0 means it will wait indefinitely for a key press
    cv2.destroyAllWindows()
