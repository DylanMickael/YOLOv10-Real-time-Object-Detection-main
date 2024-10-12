import cv2
import os
import time
from image import process_image  # Import the process_image function

video_path = 'Test/videos/crash.mp4'  # Ensure the correct file extension
cap = cv2.VideoCapture(video_path)  # Change 0 to the URL of the camera if needed

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 2)  # Calculate the number of frames to skip (5 seconds)

frame_count = 0  # Initialize frame count
processed_image = None  # Initialize processed_image with None

while True:
    ret, image = cap.read()
    if not ret:
        break

    frame_count += 1  # Increment the frame counter

    # Process the image every 5 seconds
    if frame_count % interval_frames == 0:
        processed_image, accident_detected = process_image(image)

        # Log message to indicate if an accident was detected
        if accident_detected:
            print("Accident detected! Capturing the image.")
    else:
        # If not processing, use the original image for display
        processed_image = image

    # Optionally, display the processed image (if needed)
    cv2.imshow('frame', processed_image)  # Display the processed image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
