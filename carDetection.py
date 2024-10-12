from ultralytics import YOLO
import cv2

model = YOLO('yolov10n.pt')

def detect_car(image):
    if image is None:
        print("Error: Could not load image")
        return False
    
    results = model(image)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            if class_detected_name in ['car', 'truck']:
                return True
    return False
