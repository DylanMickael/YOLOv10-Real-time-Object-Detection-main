from ultralytics import YOLO 
import cvzone
import cv2
import os
import time

model = YOLO('yolov10n.pt')

def save_image(image, class_name):
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'captures/{class_name}_{timestamp}.png'
    
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")

# cap = cv2.VideoCapture("<camera_url>")
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    results = model(image)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = box.cls[0]
            class_detected_number = int(class_detected_number)
            class_detected_name = results[0].names[class_detected_number]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
            
            if class_detected_name == 'toothbrush':
                print("Toothbrush detected!")
                save_image(image, class_detected_name)
                
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
