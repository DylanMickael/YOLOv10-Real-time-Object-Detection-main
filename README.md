Here’s a sample `README.md` for your YOLOv10 Real-time Object Detection project:

---

# YOLOv10 Real-time Object Detection

This project demonstrates real-time object detection using **YOLOv10**. It uses a webcam or an IP camera to detect objects in a live feed, highlighting the objects with bounding boxes and labels. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system.

## Features

- Real-time object detection from webcam or IP camera feed.
- Bounding boxes and class labels displayed on detected objects.
- Easily switch between local webcam and external video sources (IP camera).
- Custom detection logic, such as detecting specific objects (e.g., toothbrush).

## Requirements

Before running the project, make sure you have the following installed:

- Python 3.x
- OpenCV
- YOLO model weights
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- cvzone (for additional OpenCV enhancements)

You can install the required Python packages using the following command:

```bash
pip install ultralytics opencv-python-headless cvzone
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/YOLOv10-Real-time-Object-Detection-main.git
   cd YOLOv10-Real-time-Object-Detection-main
   ```

2. **Download the YOLOv10 model weights:**

   The code automatically downloads the weights, but you can manually download it from [Ultralytics YOLOv10n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt).

3. **Run the program:**

   For webcam:
   ```bash
   python main.py
   ```

   For IP camera (replace with your IP camera's URL):
   ```bash
   python main.py --source "http://<ip-address>:<port>/video"
   ```

## Usage

By default, the program will:

- Capture video from a connected webcam or IP camera.
- Detect objects in real-time.
- Display the object name and confidence score in the video feed.

You can modify the code to add specific actions when detecting certain objects (e.g., printing a message when a **toothbrush** is detected).

### Sample Output

When running the detection, you will see bounding boxes around detected objects, with labels indicating the object class and confidence score.

## Customization

You can customize the object detection behavior by modifying the detection logic:

- **To detect specific objects (like a toothbrush)**, modify the detection condition in `main.py`:

   ```python
   if class_detected_name == 'toothbrush':
       print("Toothbrush detected!")
   ```

- **Switch video source:** You can easily switch between webcam and IP camera by changing the video source in the `cap` initialization.

## Contributions

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements or bug fixes.

---

This `README.md` includes the necessary details about your project, such as installation instructions, usage, customization, and features. You can adjust the `git clone` URL to reflect the actual repository URL when you create it.
