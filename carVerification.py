from ultralytics import YOLO
import cv2

model = YOLO('models/yolov10n.pt')

def verify_car_in(image):
    if image is None:
        print("Error: Could not load image")
        return False
    
    results = model(image)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]
            if class_detected_name in ['car', 'cars', 'truck', 'trucks']:
                print("Car detected !")
                return True
    return False