from ultralytics import YOLO
import cv2

# Load the YOLOv10 model
modelYOLO = YOLO('models/yolov10n.pt')

# Get vocabulary from the model
vocab = modelYOLO.names

video_path = 'Test/videos/crash.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Use YOLOv10 for detection
    results_yolo = modelYOLO(image)

    # Draw detections on the image
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab[class_detected_number]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3) 
            cv2.putText(image, f'{class_detected_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check for car or truck to detect an car
            if class_detected_name in ['car']:
                print("Car detected!")
                
    # Show the processed image
    cv2.imshow('Video Stream', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
