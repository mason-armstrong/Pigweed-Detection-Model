#Loads data for training and testing
import cv2
import os
import numpy as np

#Annotation name: f"pigweed_{i:03d}.txt
#Image name: f"pigweed_{i:03d}.png
class Dataloader:
    def __init__(self, image_folder, annotation_folder, batch_size, input_shape, num_classes):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_samples = len(os.listdir(image_folder))
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_index = 0
        
    def load_images(self):
        image_names = os.listdir(self.image_folder)
        image_paths = [os.path.join(self.image_folder, image_name) for image_name in image_names]
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {image_path}")
                continue
            images.append(image)
        return images
    
    def load_annotations(self):
        annotation_names = os.listdir(self.annotation_folder)
        annotation_paths = [os.path.join(self.annotation_folder, annotation_name) for annotation_name in annotation_names]
        annotations = []
        for annotation_path in annotation_paths:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                bounding_boxes = []
                for line in lines:
                    values = line.split()
                    class_id, center_x, center_y, width, height = map(float, values)
                    bounding_boxes.append([class_id, center_x, center_y, width, height])
                annotations.append(bounding_boxes)
        return annotations   