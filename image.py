from ultralytics import YOLO
import cvzone
import cv2
import os
import time
import numpy as np

# Charger les deux modèles
model = YOLO('models/trainedYolo.pt')
modelYOLO = YOLO('models/yolov10n.pt')

# Obtenir les vocabulaires des deux modèles
vocab1 = model.names  # Returns a dictionary
vocab2 = modelYOLO.names  # Returns a dictionary

# Convert dictionaries to lists of class names
vocab1_list = list(vocab1.values())
vocab2_list = list(vocab2.values())

# Merge the vocabularies
merged_vocab = list(set(vocab1_list + vocab2_list))

# Créer les mappings pour les classes de chaque modèle vers le vocabulaire fusionné
mapping1 = {i: merged_vocab.index(name) for i, name in vocab1.items()}
mapping2 = {i: merged_vocab.index(name) for i, name in vocab2.items()}

last_capture_time = 0
capture_interval = 5
last_class_detected_name = None
last_image = None
pixel_diff_threshold = 25000

def save_image(image, class_name):
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'captures/{class_name}_{timestamp}.png'
    
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")

# Charger une image de test
image_path = 'Test/images/crash.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Appliquer le premier modèle (accident-viewer.pt)
    results = model(image)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab1[class_detected_number]  # Nom de classe original
            
            # Récupérer le nom de classe fusionné
            merged_class_name = merged_vocab[mapping1[class_detected_number]]

            # Dessiner le rectangle et le label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{merged_class_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
            
            # Logique de détection d'accident
            current_time = time.time()
            if merged_class_name in ['accident']:
                print(f"{merged_class_name} detected!")

                # Capture seulement si le délai est respecté ou nouvelle détection
                if (current_time - last_capture_time > capture_interval) or (merged_class_name != last_class_detected_name):

                    # Vérifier la différence de pixels entre les images
                    if last_image is not None:
                        diff = cv2.absdiff(last_image, image)
                        non_zero_count = np.count_nonzero(diff)

                        print(f"Pixel difference: {non_zero_count}")
                        
                        if non_zero_count > pixel_diff_threshold:
                            save_image(image, merged_class_name)
                            last_capture_time = current_time
                            last_class_detected_name = merged_class_name
                    else:
                        save_image(image, merged_class_name)
                        last_capture_time = current_time
                        last_class_detected_name = merged_class_name

                    last_image = image.copy()

    # Appliquer le second modèle (yolov10n.pt) si nécessaire
    results_yolo = modelYOLO(image)
    
    for info in results_yolo:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = vocab2[class_detected_number]
            
            # Récupérer le nom de classe fusionné
            merged_class_name = merged_vocab[mapping2[class_detected_number]]

            # Dessiner le rectangle et le label
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cvzone.putTextRect(image, f'{merged_class_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Afficher l'image avec les détections
    cv2.imshow('frame', image)
    cv2.waitKey(0)  # Attendre un appui de touche pour fermer la fenêtre

cv2.destroyAllWindows()
