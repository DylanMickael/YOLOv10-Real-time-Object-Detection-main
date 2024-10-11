from ultralytics import YOLO 
import cvzone
import cv2

model = YOLO('yolov10n.pt')
# results = model('birds.png')
# results[0].show()

cap = None

try:
    cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
    if not cap.isOpened():
        raise Exception("HTTP access error !")
except Exception as e:
    print(f"Error : {e}.")
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
                
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
