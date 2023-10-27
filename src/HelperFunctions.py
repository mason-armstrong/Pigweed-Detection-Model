import os
from keras.models import load_model
from keras.utils import plot_model
import numpy as np

#Numpy random seed
np.random.seed(42)

#Function to rename all images in a folder
def rename_images(image_folder):
    image_names = os.listdir(image_folder)
    for i, image_name in enumerate(image_names):
        os.rename(os.path.join(image_folder, image_name), os.path.join(image_folder,f"not_pigweed_{i:03d}.png"))
    print("Done renaming images")
    
    
#Function to visualize keras model
def visualize_model(model):
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    
#Function to split data into training, validation, and test. Default split is 80/10/10, change paramters to change split ratio
def split_data(image_folder, train_folder, val_folder, test_folder, train_split=0.8, val_split=0.1, test_split=0.1):
    image_names = os.listdir(image_folder)
    #shuffle list of image names
    np.random.shuffle(image_names)
    num_images = len(image_names)
    num_train = int(train_split*num_images)
    num_val = int(val_split*num_images)
    num_test = int(test_split*num_images)
    
    for i, image_name in enumerate(image_names):
        if i < num_train:
            os.rename(os.path.join(image_folder, image_name), os.path.join(train_folder, image_name))
        elif i < num_train + num_val:
            os.rename(os.path.join(image_folder, image_name), os.path.join(val_folder, image_name))
        else:
            os.rename(os.path.join(image_folder, image_name), os.path.join(test_folder, image_name))
    print("Done splitting data")