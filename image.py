from ultralytics import YOLO
import cvzone
import cv2
import os
import time
import numpy as np
import requests
import json
import geocoder
from carVerification import verify_car_in

# Load the YOLOv10 model
modelYOLO = YOLO('models/yolov10n.pt')

# Get vocabulary from the model
vocab2 = modelYOLO.names
vocab2_list = list(vocab2.values())

last_capture_time = 0
capture_interval = 20
last_class_detected_name = None
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
        send_image_with_metadata(original_filename, server_url)

def process_image(image):
    global last_capture_time, last_class_detected_name, last_image

    # Save a copy of the original image before modifications
    original_image = image.copy()

    # Use only YOLOv10 for detection
    results_yolo = modelYOLO(image)
    
    accident_detected = False
    
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab2[class_detected_number]

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cvzone.putTextRect(image, f'{class_detected_name} {confidence}%', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

            if class_detected_name in ['car', 'truck']:
                print("Accident detected!")
                accident_detected = True

                current_time = time.time()
                if (current_time - last_capture_time > capture_interval) or (class_detected_name != last_class_detected_name):

                    if last_image is not None:
                        diff = cv2.absdiff(last_image, image)
                        non_zero_count = np.count_nonzero(diff)

                        print(f"Pixel difference: {non_zero_count}")
                        
                        if non_zero_count > pixel_diff_threshold:
                            last_capture_time = current_time
                            last_class_detected_name = class_detected_name
                            last_image = image.copy()
                            save_image(image, original_image, class_detected_name)
                    else:
                        last_capture_time = current_time
                        last_class_detected_name = class_detected_name
                        last_image = image.copy()
                        save_image(image, original_image, class_detected_name)

    return image, accident_detected

# Uncomment below lines to test with an image file
# image_path = 'Test/images/crash.jpg'
# image = cv2.imread(image_path)

# if image is None:
#     print(f"Error: Could not load image from {image_path}")
# else:
#     processed_image, accident_detected = process_image(image)
# cv2.destroyAllWindows()
