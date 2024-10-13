import cv2
import os
import time
from image import process_image

video_path = 'Test/videos/crash.mp4'  # Ensure the correct file extension
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 2)

frame_count = 0
processed_image = None

while True:
    ret, image = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % interval_frames == 0:
        processed_image, accident_detected = process_image(image)

        if accident_detected:
            print("Accident detected! Capturing the image.")
    else:
        processed_image = image

    cv2.imshow('frame', processed_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
