import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(42)

class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()
        
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def summary(self):
        self.model.summary()
        
    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #Freeze layers of base model
        for layer in base_model.layers:
            layer.trainable = False
            
        #Add custom layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x) #Assumes 2 classes (pigweed or not pigweed)
        model = keras.Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])