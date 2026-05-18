from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from models.keras_model_adapters.resnet_adapter import ResNet_Interface

class TINY_IMAGENET_Dataset:
    def __init__(self, validation_size=3500, fitness_size=3500, seed=0, normalize=True, subtract_mean=True, path=None):
        # Tensorflow does not give us y in one-hot encoding, so we need to convert it ourselves
        self.n_classes = 200
        self.validation_size = validation_size
        self.fitness_size = fitness_size
        self.seed = seed
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.img_rows, self.img_cols, self.channels = 64, 64, 3
        if path is None:
            from sge.parameters import params
            self.path = os.path.join(params['DATA_DIR'], 'tiny_imagenet')
        else:
            self.path = path


    def load_data_for_evolution(self):
        # TINY_IMAGENET dataset is not available in Keras, so we use our own implementation to load it.

        # Load the TINY_IMAGENET dataset, split it into training, validation and fitness sets, and preprocess when applicable.
        # TINY_IMAGENET test set is not used in the evolution, it is only used in the final benchmark evaluation, so we can ignore it here.
        (x, y), (_, _) = self.load_data()

        # Preprocess the data
        x = x.astype('float32')

        # Normalize the data to [0, 1] range if specified in the parameters
        if self.normalize:
            x /= 255

        # Ensure data follows the correct shape for Keras
        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], self.channels, self.img_rows, self.img_cols)
        else:
            x = x.reshape(x.shape[0], self.img_rows, self.img_cols, self.channels)
        
        # We do not do one-hot encoding as it is handled inside "self.load_data"

        # Split the data into training, validation and fitness sets using stratified sampling to maintain class distribution
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                        test_size=self.validation_size + self.fitness_size,
                                                        stratify=y,
                                                        random_state=self.seed)
        x_val, x_fit, y_val, y_fit = train_test_split(x_val, y_val,
                                                        test_size=self.fitness_size,
                                                        stratify=y_val,
                                                        random_state=self.seed)

        # Subtract the mean image from the data if specified in the parameters
        if self.subtract_mean:
            x_mean = 0
            for x in x_train:
                x_mean += x
            x_mean /= len(x_train)
            x_train -= x_mean
            x_val -= x_mean
            x_fit -= x_mean

        #len(x_train) 245224 len(x_val) 7000 len(x_fit) 3000
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_fit = x_fit
        self.y_fit = y_fit
    
    def load_data_for_benchmark(self):
        # Load the TINY_IMAGENET dataset, split it into training, validation sets.
        # Load the test set of TINY_IMAGENET, separately.
        # Preprocess when applicable.
        (x, y), (x_test, y_test) = self.load_data()

        # Preprocess the data
        x = x.astype('float32')
        x_test = x_test.astype('float32')

        # Normalize the data to [0, 1] range if specified in the parameters
        if self.normalize:
            x /= 255
            x_test /= 255

        # Ensure data follows the correct shape for Keras
        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], self.channels, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], self.channels, self.img_rows, self.img_cols)
        else:
            x = x.reshape(x.shape[0], self.img_rows, self.img_cols, self.channels)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.channels)
        
        # We do not do one-hot encoding as it is handled inside "self.load_data"

        # Split the data into training, validation sets using stratified sampling to maintain class distribution
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                        test_size=self.validation_size,
                                                        stratify=y,
                                                        random_state=self.seed)

        # Subtract the mean image from the data if specified in the parameters
        if self.subtract_mean:
            x_mean = 0
            for x in x_train:
                x_mean += x
            x_mean /= len(x_train)
            x_train -= x_mean
            x_val -= x_mean
            x_test -= x_mean

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
    
    def load_data(self):
        # This expects the TINY_IMAGENET dataset to be in a data directory, in a folder called tiny_imagenet, with the following structure:
        # By default, self.path = 'data/tiny_imagenet'
        # tiny_imagenet/train/class_x/xxx.JPEG
        # tiny_imagenet/val/class_x/xxx.JPEG
        # where class_x is the name of the class, and xxx.JPEG is the name
        # The directory should be specified using self.path which is set in the constructor.

        training_path = os.path.join(self.path, 'train')
        validation_path = os.path.join(self.path, 'val')


        # We need to load from two directories so let's create a helper function to load the data from a directory
        def load_data_from_directory(path):
            # Get the list of class directories
            class_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            # Initialize lists to store data
            x = []
            y = []
            class_index = {}

            # Loop through each class directory
            for i, class_dir in enumerate(class_dirs):
                class_index[class_dir] = i
                class_path = os.path.join(path, class_dir)
                # Get the list of image files in the class directory
                image_files = [f for f in os.listdir(class_path) if f.endswith('.JPEG')]
                
                # Loop through each image file and store the data
                for image_file in image_files:
                    # Load image file and convert to numpy array
                    image_path = os.path.join(class_path, image_file)
                    image = plt.imread(image_path)
                    x.append(image)

                    # Turn class into binary vector where class_index[class_dir] is 1 and the rest are 0

                    y_vector = np.zeros(len(class_dirs))
                    y_vector[class_index[class_dir]] = 1
                    y.append(y_vector)

            # Create a numpy data set from the collected data
            return np.array(x), np.array(y)
        
        # I could not find labelled test set for this dataset.
        # We use "TINY_IMAGENET Training" for evolution data: training, validation and fitness sets.
        # We use "TINY_IMAGENET Validation" for benchmark data: test set.

        x, y = load_data_from_directory(training_path)
        x_test, y_test = load_data_from_directory(validation_path)

        # Return in the same form as keras datasets to keep it neat and consistent with the rest of the codebase
        return (x, y), (x_test, y_test)
