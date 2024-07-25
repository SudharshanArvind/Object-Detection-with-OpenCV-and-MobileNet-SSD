# Object-Detection-with-OpenCV-and-MobileNet-SSD
This project demonstrates a real-time object detection system using OpenCV and a pre-trained MobileNet-SSD model with the COCO dataset. The system captures video from a webcam, processes each frame to detect objects, and displays the detection results with bounding boxes and class labels.

This repository uses pre-trained MobileNet-SSD v3 model for Object Detection: https://github.com/chuanqi305/MobileNet-SSD

The model uses COCO dataset for training: https://cocodataset.org/#overview

The COCO dataset consists of 80 classes of images so the objects detected would belong from one of these classes only.


# Features
Real-time object detection using a webcam.

Uses MobileNet-SSD, a lightweight and efficient object detection model.

Classifies objects into one of the 80 categories defined in the COCO dataset.

Displays bounding boxes and class labels for detected objects.

Adjustable confidence threshold for detections.


# Requirements
Python 3.x

OpenCV

Pre-trained MobileNet-SSD model files ('.pb' and '.pbtxt')

COCO class labels file ('coco.names')


# Required Libraries
pip install opencv-python

# Download the required model files:
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt (model configuration)

frozen_inference_graph.pb (pre-trained model weights)
coco.names (COCO class labels)
