from ultralytics import YOLO
import cv2
import os
import time
import requests
import json
import geocoder

# Load the YOLOv10 model
modelYOLO = YOLO('models/yolov10n.pt')

# Get vocabulary from the model
vocab2 = modelYOLO.names

last_capture_time = 0
capture_interval = 30
last_image = None
pixel_diff_threshold = 100000

def get_phone_location():
    g = geocoder.ip('102.19.119.251')
    if g.ok:
        return g.latlng
    else:
        print("Impossible de récupérer la localisation")
        return None, None

def send_image_with_metadata(image_path, server_url):
    latitude, longitude = get_phone_location()

    with open(image_path, 'rb') as image_file:
        metadata = {
            'userId': 14,
            'typeId': 1,
            'latitude': latitude,
            'longitude': longitude,
            'state': "PENDING",
            'description': 'Accident détecté par Caméra'
        }

        files = {'assets': ('image.png', image_file, 'image/png')}
        data = {'signal': json.dumps(metadata)}

        try:
            response = requests.post(server_url, files=files, data=data, timeout=20)

            if response.status_code == 200:
                print(f"Image et métadonnées envoyées avec succès à {server_url}")
            else:
                print(f"Échec de l'envoi : {response.status_code}")

        except requests.exceptions.Timeout:
            print("Erreur lors de l'envoi : Timeout atteint. Le serveur n'a pas répondu à temps.")

        except requests.exceptions.ConnectionError:
            print("Erreur lors de l'envoi : Impossible de se connecter au serveur.")

        except Exception as e:
            print(f"Erreur lors de l'envoi : {e}")

def save_image(image, original_image, class_name):
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    modified_filename = f'captures/{class_name}_{timestamp}_modified.png'
    original_filename = f'captures/{class_name}_{timestamp}_original.png'
    
    # Save the modified image
    cv2.imwrite(modified_filename, image)
    print(f"Image modifiée sauvegardée : {modified_filename}")
    
    # Save the original image
    cv2.imwrite(original_filename, original_image)
    print(f"Image originale sauvegardée : {original_filename}")
    
    # URL du serveur
    server_url = 'http://192.168.117.193:8080/api/signal'
    
    # Send the original image if it is an accident
    if class_name == 'car':
        send_image_with_metadata(modified_filename, server_url)


# Capture video from the camera or video stream
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    original_image = image.copy()
    
    # Use YOLOv10 for detection
    results_yolo = modelYOLO(image)

    accident_detected = False  # Flag to check if an accident is detected

    # Draw detections on the image
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab2[class_detected_number]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3) 
            cv2.putText(image, f'{class_detected_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check for car or truck to detect an accident
            if class_detected_name in ['car', 'truck']:
                accident_detected = True
                print("Accident detected!")
                
                # Draw red rectangle and label for accident

                # Save the image if an accident is detected
                if (time.time() - last_capture_time > capture_interval) or (last_image is None):
                    save_image(image, original_image, class_detected_name)
                    last_capture_time = time.time()
                    last_image = image.copy()

    # Show the processed image
    cv2.imshow('Video Stream', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
