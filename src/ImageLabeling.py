# File to label and classify images for yolo detection model

import os
import labelImg
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from ultralytics import YOLO


imgs_path = 'YOLOData\\test\\images'
best_model = 'runs\\detect\\train\\weights\\best.pt'
yolov8_path = 'yolov8x.pt'

data_path = "YOLOData\pigweed_data.yaml"

