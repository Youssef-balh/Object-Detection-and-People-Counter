# Object Detection and Tracking with YOLOv8

## Overview

This project implements object detection and tracking using YOLOv8 with pre-trained weights on the COCO dataset. The system also features a people counter that utilizes a sorting algorithm for tracking. The graphical interface is built with Streamlit, and the project supports detection on videos, images, and real-time scenarios.

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
   
