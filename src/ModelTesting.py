from keras.models import load_model
from DataPreProcessing import DataPreProcessing
import cv2


class ModelTesting:
    def __init__(self, model, test_folder):
        self.model = load_model(model)
        self.test_data = DataPreProcessing(test_folder, 12, (224, 224), 'binary')
        
        
    def test(self):
        loss, accuracy = self.model.evaluate(self.test_data.generator)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        
    def predict(self):
        image_batch, _ = next(iter(self.test_data.generator))
        #Transform image for predicition
        predictions = self.model.predict(image_batch)
        
        return predictions, image_batch
    
    def individual_prediction(self, image):
        prediction = self.model.predict(image)
        #Print loss and accuarcy
        return prediction
    
    
    
if __name__ == '__main__':
    
    test = ModelTesting('my_model.keras', 'data/processed/testing')
    
    for i, (image_batch, label_batch) in enumerate(test.test_data.generator):
        for j, single_image in enumerate(image_batch):
            single_image = single_image.reshape(1, 224, 224, 3)
            prediction = test.individual_prediction(single_image)
            if prediction > 0.5:
                print("Pigweed")
                text = "Pigweed"
            else:
                print("Not Pigweed")
                text = "Not Pigweed"

            # Add text to the image
            cv2.putText(single_image[0],                     # Image
            text,                              # Text to add
            (10, 50),                          # Bottom-left corner coordinates
            cv2.FONT_HERSHEY_SIMPLEX,          # Font
            1,                                 # Font scale
            (255, 0, 0),                       # Font color (BGR)
            2)                                 # Line thickness


            import sys

            cv2.setWindowTitle(f"Image {j}", "Not Pigweed")            
            cv2.setWindowTitle(f"Image {j}", test.test_data.generator.filenames[j])
            cv2.imshow(f"Image {j}", (single_image[0] * 255).astype('uint8'))
            cv2.resizeWindow(f"Image {j}", 1200, 800)
            #Fill window with easy to look at colors
            cv2.setWindowProperty(f"Image {j}", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.waitKey(0)
            sys.exit()
      

