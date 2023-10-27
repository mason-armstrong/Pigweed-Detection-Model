import os
import glob
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
from PIL import Image
import torch
from torchvision import transforms
import pathlib


imgs_path = 'YOLOData\\test\\images'
best_model = 'runs\\detect\\train5\\weights\\best.pt'
yolov8_path = 'yolov8x.pt'
yolov5_path = 'yolov5su.pt'

data_path = "YOLOData\pigweed_data.yaml"
model = YOLO(best_model)  # initialize


#model.train(data=data_path, imgsz = 640, epochs = 2, batch = -1, pretrained = True, model=yolov5_path)





results = model.predict(imgs_path, conf=0.5)
for result in results:
    boxes = result.boxes
    masks = result.masks
    probs = result.probs
    im_array = result.plot()
    im = Image.fromarray(im_array[...,::-1])
    im.show()
    im.save('results.jpg')

    
