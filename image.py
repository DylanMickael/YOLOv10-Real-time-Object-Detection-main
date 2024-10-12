from ultralytics import YOLO
import cvzone
import cv2
import os
import time
import numpy as np
import requests
import json
import geocoder

model = YOLO('models/trainedYolo.pt')
modelYOLO = YOLO('models/yolov10n.pt')

vocab1 = model.names
vocab2 = modelYOLO.names

vocab1_list = list(vocab1.values())
vocab2_list = list(vocab2.values())

merged_vocab = list(set(vocab1_list + vocab2_list))

mapping1 = {i: merged_vocab.index(name) for i, name in vocab1.items()}
mapping2 = {i: merged_vocab.index(name) for i, name in vocab2.items()}

last_capture_time = 0
capture_interval = 2
last_class_detected_name = None
last_image = None
pixel_diff_threshold = 5000000

def get_phone_location():
    g = geocoder.ip('102.19.119.251')
    if g.ok:
        return g.latlng
    else:
        print("Impossible de récupérer la localisation")
        return None, None
    
def send_image_with_metadata(image_path, class_name, accident_timestamp, server_url):
    latitude, longitude = get_phone_location()  
    print(latitude, longitude)
    
    with open(image_path, 'rb') as image_file:
        metadata = {
            'userId': 1, 
            'typeId': 1,
            'latitude': latitude,
            'longitude': longitude,
            'description': f'Accident détecté par Caméra : {class_name}',
            'timestamp': accident_timestamp
        }

        files = {'assets': image_file}
        data = {'signal': json.dumps(metadata)}

        try:
            response = requests.post(server_url, files=files, data=data)
            if response.status_code == 200:
                print(f"Image et métadonnées envoyées avec succès à {server_url}")
            else:
                print(f"Échec de l'envoi : {response.status_code}")
        except Exception as e:
            print(f"Erreur lors de l'envoi : {e}")

def save_image(image, class_name):
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'captures/{class_name}_{timestamp}.png'
    
    cv2.imwrite(filename, image)
    print(f"Image sauvegardée : {filename}")
    
    server_url = 'http://localhost:8080/api/signal'
    
    if class_name == 'accident':
        accident_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        send_image_with_metadata(filename, class_name, accident_timestamp, server_url)

def process_image(image):
    global last_capture_time, last_class_detected_name, last_image

    results = model(image)
    
    accident_detected = False
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab1[class_detected_number]
            
            merged_class_name = merged_vocab[mapping1[class_detected_number]]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{merged_class_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
            
            current_time = time.time()
            if merged_class_name in ['accident']:
                print(f"{merged_class_name} detected!")
                accident_detected = True 

                if (current_time - last_capture_time > capture_interval) or (merged_class_name != last_class_detected_name):

                    if last_image is not None:
                        diff = cv2.absdiff(last_image, image)
                        non_zero_count = np.count_nonzero(diff)

                        print(f"Pixel difference: {non_zero_count}")
                        
                        if non_zero_count > pixel_diff_threshold:
                            last_capture_time = current_time
                            last_class_detected_name = merged_class_name
                            last_image = image.copy()
                            save_image(image, merged_class_name)
                    else:
                        last_capture_time = current_time
                        last_class_detected_name = merged_class_name
                        last_image = image.copy()
                        save_image(image, merged_class_name)

    results_yolo = modelYOLO(image)
    
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab2[class_detected_number]
            
            merged_class_name = merged_vocab[mapping2[class_detected_number]]

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cvzone.putTextRect(image, f'{merged_class_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    return image, accident_detected

# image_path = 'Test/images/crash.jpg'
# image = cv2.imread(image_path)

# if image is None:
#     print(f"Error: Could not load image from {image_path}")
# else:
#     processed_image, accident_detected = process_image(image)
# cv2.destroyAllWindows()
