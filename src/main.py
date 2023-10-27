from CNNModel import CNNModel
from keras.utils import load_model
import logging
from keras.callbacks import TensorBoard
from ModelTraining import ModelTraining
import tensorflow as tf




if __name__ == '__main__':
    logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
    
    file_writer = tf.summary.create_file_writer('logs/')
    
    #Paths to data
    val_folder = 'data/processed/validated'
    train_folder = 'data/processed/training'
    test_folder = 'data/processed/testing'
   
    
    cnn_model = CNNModel((224, 224, 3), 1)
    cnn_model.compile_model()
    model_trainer = ModelTraining(cnn_model, train_folder, val_folder, 8, 12, tensorboard)
    # Log the first batch of training images
    image_batch, _ = next(iter(model_trainer.train_data.generator))
    image_batch = (image_batch * 255).astype("uint8")
    with file_writer.as_default():
        tf.summary.image("Training Data", image_batch, step=0,max_outputs=12)

    cnn_model.model.fit(
        model_trainer.train_data.generator,
        epochs=2,
        validation_data=model_trainer.val_data.generator,
        callbacks=[tensorboard]
        )

        
    cnn_model.model.save('my_model.keras')