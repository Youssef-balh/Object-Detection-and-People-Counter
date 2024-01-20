import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st
from sort import *
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image


model = YOLO("../Yolo-Weights/yolov8l.pt")
# Define the COCO class labels
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "tomato", "sandwich", "orange", "broccoli",
              "carrot", "Hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Function to display information about the YOLO model and its variants
def about_app():
    # Centered title using HTML and Markdown
    centered_text = '<h1 style="color:#001B79; text-align:center;">What is YOLO Model?</h1>'
    st.markdown(centered_text, unsafe_allow_html=True)
    
    # Information about YOLOv8 and its features
    st.markdown('<h1 style="color:#2B3499;">What is Yolo Model ?</h1>', unsafe_allow_html=True)
    st.write("##")
    st.write(
        """
        YOLOv8 is based on a deep convolutional neural network (CNN) architecture that is similar to its predecessors.
        However, it introduces a number of new features and improvements, including:
        - A new backbone architecture called CSPNet, which is more efficient and accurate than previous backbones.
        - A new neck architecture called FPN+PAN, which better aggregates features from different levels of the backbone.
        - A new head architecture called PANet, which is more robust to occlusion and scale variations.
        - A new training procedure that uses a combination of supervised and unsupervised learning.
        """
    )
    st.image('plotYolo1.png')  # Display an image related to YOLOv8
    
    # Information about the COCO dataset
    st.markdown('<h1 style="color:#2B3499;">Yolo Model Use The CoCo Dataset</h1>', unsafe_allow_html=True)
    st.write("##")
    st.write(
        """
        - COCO (Common Objects in Context) is the industry standard benchmark for evaluating object detection models.
        - When comparing models on COCO, we look at the mAP value and FPS measurement for inference speed.
        - COCO accuracy is state of the art for models at comparable inference latencies.
        - YOLOv8 Detect, Segment and Pose models pretrained on the COCO dataset.
        """
    )
    st.image('yoloimg.png', width=700)  # Display an image related to the COCO dataset
    
    # Information about different versions of YOLOv8
    st.markdown('<h1 style="color:#2B3499;">Diffrence Version of Yolov8</h1>', unsafe_allow_html=True)
    st.write("##")
    st.write(
        """
        Difference between variants of Yolo V8:
        - YOLOv8 is available in three variants: YOLOv8, YOLOv8-L, and YOLOv8-X.
        - The main difference between the variants is the size of the backbone network.
        - YOLOv8 has the smallest backbone network, while YOLOv8-X has the largest backbone network.
        - The larger backbone network in YOLOv8-X gives it better accuracy, but it also makes it slower than YOLOv8 and YOLOv8-L.
        """
    )
    st.image('yoloDiffrece.jpeg', width=700)  # Display an image showing the differences between YOLOv8 variants
    
    

# Function for real-time object detection with the webcam
def Yolo_webcam():
    # Display the header and introduction
    st.header('Object detection with the webcam:')
    st.markdown('<p style="font-size: 25px;">The provided code utilizes the OpenCV and YOLO (You Only Look Once) frameworks '
                'to implement real-time object detection through a webcam feed. It initializes the webcam with specified '
                'dimensions, loads a pre-trained YOLO model, and defines a list of object classes.</p>', unsafe_allow_html=True)
    
    # Initialize the webcam with specified dimensions
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Display a sidebar for additional information
    st.sidebar.write("Object detection with the webcam")

    # Display button to start/stop the camera
    btn_start_camera = st.button("Activate camera")

    if btn_start_camera:
        # Warning message when the camera is activated
        st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")
        camera_active = True
        btn_stop_camera = st.button("Stop camera")
        
        # Initialize placeholders for image and text display
        img_placeholder = st.empty()
        text_placeholder = st.empty()
        
        # Main loop for real-time object detection
        while camera_active:
            success, img = cap.read()
            results = model(img, stream=True)
            detected_objects = []

            # Iterate through detected objects and draw rectangles
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 0), colorR=(0, 0, 255))

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    # Add detected objects to the list if confidence is above 0.7
                    if conf > 0.7:
                        detected_objects.append(f"{classNames[cls]}: {conf}")

            # Display the detected objects if any, otherwise, clear the text placeholder
            if detected_objects:
                text_placeholder.markdown(
                    '<div style="background-color: #ECE3CE; padding: 10px; margin-bottom : 10px;border-radius: 5px; '
                    'font-size: 30px;">'
                    '<b>{}</b></div>'.format("<br>".join(detected_objects)),
                    unsafe_allow_html=True)
            else:
                text_placeholder.empty()
            
            # Display the webcam image with detected objects
            img_placeholder.image(img, channels="BGR", use_column_width=True, output_format="BGR",
                                   caption="Real-time Webcam Image")

            # Check if the stop button is pressed
            if btn_stop_camera:
                camera_active = False
                break  

        # Release the webcam when the loop is terminated
        cap.release()

# Function for real-time people counting with the webcam
def counter(): 
    # Display the header and introduction
    st.header('People counter')
    st.markdown('<p style="font-size: 25px;">This system provides an efficient and accurate solution for real-time people counting '
                'and detection in various applications such as surveillance.</p>', unsafe_allow_html=True)
     
    # Initialize the webcam with specified dimensions
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Display a sidebar for additional information
    st.sidebar.write("Détection d'objets avec la webcam")

    # Display button to start/stop the camera
    btn_start_camera = st.button("Activer la caméra")

    if btn_start_camera:
        # Warning message when the camera is activated
        st.warning("La caméra est activée. Pour arrêter, cliquez sur le bouton 'Arrêter la caméra'.")
        camera_active = True
        btn_stop_camera = st.button("Stop camera")
        img_placeholder = st.empty()
        text_holder = st.empty()
        
        # Initialize object tracker (Assuming "sort.py" is available)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        # Main loop for real-time people counting
        while camera_active:
            success, img = cap.read()
            results = model(img, stream=True)

            detections = np.empty((0, 5))

            # Iterate through detected objects and draw rectangles
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    # Draw bounding box and label if confidence is above 0.5
                    if conf > 0.5:
                        cvzone.putTextRect(img, f'{classNames[cls]}{conf}', pos=(max(0, x1), max(35, y1)), scale=1, thickness=1)

                    # Check if the detected object is a person and confidence is above 0.5
                    if currentClass == "person" and conf > 0.5:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

            # Update the object tracker with detections
            resultsTracker = tracker.update(detections)

            # Draw rectangles for tracked people
            for result in resultsTracker:
                x1, y1, x2, y2, _ = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate and display the total count of people
            total_count = len(resultsTracker)
            text_holder.markdown(
                f'<div style="background-color: #ECE3CE; padding: 10px; border-radius: 5px; font-size: 18px;">'
                f'<b>People Total count:</b> {total_count}</div>',
                unsafe_allow_html=True
            )

            # Display the webcam image with detected people
            img_placeholder.image(img, channels="BGR", use_column_width=True, output_format="BGR",
                                  caption="Real-time Webcam Image")

            # Check if the stop button is pressed
            if btn_stop_camera:
                camera_active = False
                break  
        
        # Release the webcam when the loop is terminated
        cap.release()


# Function for image and video object detection
def yolo_video_img():
    # Display the header and introduction
    st.header('Image and Video Object Detection')
    st.markdown('<p style="font-size: 25px;">Explore our object detection interface for images and videos, choose between "Image" '
                'or "Video" to discover the powerful capabilities of our system in identifying and highlighting objects.</p>',
                unsafe_allow_html=True)

    # Select the mode (Image or Video) and input the path
    txt_mode = st.selectbox("Choisissez", ["Image", "Vidéo"])
    text = st.text_input("Enter the path of the image or video :")

    # Button to trigger the object detection
    if st.button("Convertir"):
        if txt_mode == "Image":
            # Perform object detection on the image
            detect_objects_in_image(text)
        else:
            # Perform object detection on the video
            detect_objects_in_video(text)

# Function to detect objects in an image
def detect_objects_in_image(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Perform object detection
    place_img = st.empty()
    results = model(img)
    place = st.empty()
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.7:
                detected_objects.append(f"{classNames[cls]}: {conf}")

            # Draw bounding box and label
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

    place_img.image(img, channels="BGR", use_column_width=True, output_format="BGR", caption="Annotated Image")

    if detected_objects:
        place.markdown(
            '<div style="background-color: #ECE3CE; padding: 10px; margin-bottom : 10px;border-radius: 5px; font-size: 30px;">'
            '<b>{}</b></div>'.format("<br>".join(detected_objects)),
            unsafe_allow_html=True)
    else:
        place.empty()

# Function to detect objects in a video
def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)  # For Video
    prev_frame_time = 0
    new_frame_time = 0
    place = st.empty()
    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)
        place.image(img, channels="BGR", use_column_width=True, output_format="BGR", caption="Real-time Webcam Image")

def main():
    # Display a welcome message at the center of the page
    st.markdown("<h1 style='text-align: center;margin-bottom:20px;'>Welcome to our simulation.</h1>", unsafe_allow_html=True)

    # Display an image in the sidebar
    st.sidebar.image('imgObj1.png')

    # Select the app mode from the sidebar
    app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Image and Video Object Detection',
                                                            'Object detection using webcam', 'People counter'])

    # Determine the selected app mode and call the corresponding function
    if app_mode == 'About App':
        about_app()
    elif app_mode == 'Image and Video Object Detection':
        yolo_video_img()
    elif app_mode == 'Object detection using webcam':
        Yolo_webcam()
    else:
        counter()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
