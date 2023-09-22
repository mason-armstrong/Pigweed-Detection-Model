from keras.preprocessing.image import ImageDataGenerator

class DataPreProcessing:
    """
    A class for setting up data generators for image classification tasks.
    
    Attributes:
    -----------
    image_folder : str
        The path to the directory containing the images.
    batch_size : int
        The number of samples per batch.
    target_size : tuple
        The dimensions to which all images found will be resized.
    class_mode : str
        One of "categorical", "binary", "sparse", "input", or None. Determines the type of label arrays that are returned.
    """
    def __init__(self, image_folder, batch_size, target_size, class_mode):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        
        self.data_gen = self.setup_data_gen()
        self.generator = self.setup_generator()
        
        self.num_batches = self.generator.n // self.batch_size
    def setup_data_gen(self):
        """
        Sets up an ImageDataGenerator object with specified augmentation parameters.
        
        Returns:
        --------
        A configured ImageDataGenerator object.
        """
        data_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        return data_gen
    
    def setup_generator(self):
        """
        Sets up a directory iterator for loading images and their labels.
        
        Returns:
        --------
        A configured directory iterator.
        """
        generator = self.data_gen.flow_from_directory(
            self.image_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode
        )
        return generator
    
    
    def get_batch(self):
        """
        Returns a batch of images and their labels.
        
        Returns:
        --------
        A tuple containing a batch of images and their labels.
        """
        return self.generator.next()
    

            

