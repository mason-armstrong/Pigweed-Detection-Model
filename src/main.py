from DataPreProcessing import DataPreProcessing
from Dataloader import Dataloader
from CNNModel import CNNModel
import os



if __name__ == '__main__':
    #Paths to data
    image_folder = 'data/raw/images'
    print(f"Image folder: {image_folder}")
    print(os.listdir(image_folder))
    
    #Check if folders contain same number of files
    image_names = os.listdir(image_folder)
    #Check if folders contain anything at all
    assert len(image_names) > 0, "No images found"
    
    
    #Initializations
    preprocessor = DataPreProcessing(image_folder, 32, (224, 224), 'binary')
    cnn_model = CNNModel((224, 224, 3), 1)
    
    #compile model
    cnn_model.compile_model()
    
    #Training loop
    for epoch in range(10):
        for batch in range(preprocessor.num_batches):
            batch_images, batch_annotations = preprocessor.get_batch()
            loss = cnn_model.model.train_on_batch(batch_images, batch_annotations)
            print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss}")

            
    cnn_model.model.save('my_model.keras')