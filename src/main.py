from DataPreProcessing import DataPreProcessing
from Dataloader import Dataloader
from CNNModel import CNNModel
from keras.utils import plot_model
import logging
from keras.callbacks import TensorBoard
import os



logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
if __name__ == '__main__':
    #Paths to data
    val_folder = 'data/processed/validated'
    train_folder = 'data/processed/training'
    test_folder = 'data/processed/testing'

        
    #Initialize DataPreProcessing
    train_data = DataPreProcessing(train_folder, 32, (224, 224), 'binary')
    val_data = DataPreProcessing(val_folder, 32, (224, 224), 'binary')
    

    cnn_model = CNNModel((224, 224, 3), 1)
    cnn_model.compile_model()
    
    #Training loop
    logging.info(f"Starting training...")
    for epoch in range(32):
        for batch in range(train_data.num_batches):
            batch_images, batch_annotations = train_data.get_batch()
            loss = cnn_model.model.train_on_batch(batch_images, batch_annotations)
            logging.info(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss}")
            
            #Log data for tensorboard
            tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
            

        #Validation loop
        epoch_val_loss = 0
        for batch in range(val_data.num_batches):
            batch_images, batch_annotations = val_data.get_batch()
            val_loss = cnn_model.model.evaluate(batch_images, batch_annotations, verbose=0)
            epoch_val_loss += val_loss[0]  # Assuming loss is the first item in the returned list
        
    #Calculate average validation loss for this epoch
    epoch_val_loss /= val_data.num_batches
    logging.info(f"Epoch: {epoch}, Validation Loss: {epoch_val_loss}")
    
    #Log training and validation loss to a file
    with open('training_log.txt', 'a') as f:
        f.write(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {epoch_val_loss}\n")
        
    
        

            
    cnn_model.model.save('my_model.keras')