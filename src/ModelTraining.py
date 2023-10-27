from DataPreProcessing import DataPreProcessing
import logging


class ModelTraining:
    def __init__(self, model, train_folder, val_folder, epochs, batch_size, tensorboard):
        self.model = model
        self.train_data = DataPreProcessing(train_folder, batch_size, (224, 224), 'binary')
        self.val_data = DataPreProcessing(val_folder, batch_size, (224, 224), 'binary')
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.tensorboard = tensorboard
        
        
    def train_one_epoch(self):
        logs = {}
        for batch in range(self.train_data.num_batches):
            batch_images, batch_annotations = self.train_data.get_batch()
            loss = self.model.model.train_on_batch(batch_images, batch_annotations)
            logs['loss'] = loss
            self.tensorboard.on_epoch_end(batch, logs)
            logging.info(f"Epoch: {self.epoch}, Batch: {batch}, Loss: {loss}")
        return loss
            
            
            
    def validate_one_epoch(self):
        logs = {}
        epoch_val_loss = 0
        for batch in range(self.val_data.num_batches):
            batch_images, batch_annotations = self.val_data.get_batch()
            val_loss = self.model.evaluate(batch_images, batch_annotations, verbose=0)
            epoch_val_loss += val_loss[0]  
            # Assuming loss is the first item in the returned list
        logs['val_logs'] = epoch_val_loss / self.val_data.num_batches
        self.tensorboard.on_epoch_end(self.epochs, logs)
        logging.info(f"Epoch: {self.epochs}, Validation Loss: {epoch_val_loss}")
        return epoch_val_loss
            
        
    def train(self):
        logging.info("Starting Training...")
        for epoch in range(self.epochs):
            loss = self.train_one_epoch()
            epoch_val_loss = self.validate_one_epoch()
        #Log training data and validation data to a file
        with open('training_log.txt', 'a') as f:
            f.write(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {epoch_val_loss}\n")