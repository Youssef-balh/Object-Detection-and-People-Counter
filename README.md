# Object Detection and Tracking with YOLOv8

## Overview

This project implements object detection and tracking using YOLOv8 with pre-trained weights on the COCO dataset. The system also features a people counter that utilizes a sorting algorithm for tracking. The graphical interface is built with Streamlit, and the project supports detection on videos, images, and real-time scenarios.For each 
detected object we draw bound boxes arround objects with confidence level above 0.6 (60%).

## Why Sorting Algorithm for Object Tracking 
The choice between using a sorting algorithm for object tracking and a simple incrementing method hinges on the complexity of the tracking task. Sorting algorithms, such as SORT, are advantageous in scenarios with occlusions, dynamic movements, and complex object interactions. They excel at associating and matching object IDs across frames, providing accurate tracking even in challenging conditions. On the other hand, a simple incrementing method is more straightforward and computationally efficient, making it suitable for less complex tracking tasks or controlled environments where objects move predictably and occlusions are minimal. 

## Project Structure

The project is organized as follows:

- **`main.py`**: The main script to execute object detection and tracking.
- **`sort.py`**: File containing the sorting algorithm for tracking.
- **`yolo-weights/`**: Directory for storing YOLOv8 pre-trained weights (`yolov8.pt`).

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/object-detection-tracking.git
   cd object-detection-tracking
2. **Install Dependencies:**
   - **streamlit**
   - **numpy**
   - **Pillow (PIL)**
   - **pandas**
   - **ultralytics (YOLO)**
   - **opencv (cv2)**
   - **cvzone**
   - **math**
3. **Run**
![Image 1](https://raw.githubusercontent.com/Youssef-balh/Object-Detection-and-People-Counter/main/assets/113738047/7268954e-7eb0-40b9-a603-eaebb74b5ad3) | ![Image 2](https://github.com/Youssef-balh/Object-Detection-and-People-Counter/assets/113738047/a3c54f86-4e76-41cf-b313-126498b4afb5)






