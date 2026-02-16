from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist as fmnist
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras

class FMNIST_Dataset:
    def __init__(self, validation_size=3500, fitness_size=3500, seed=0, normalize=True, subtract_mean=True):
        # Tensorflow does not give us y in one-hot encoding, so we need to convert it ourselves
        self.n_classes = 10
        self.validation_size = validation_size
        self.fitness_size = fitness_size
        self.seed = seed
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.img_rows, self.img_cols, self.channels = 28, 28, 1

    def load_data_for_evolution(self):
        # Load the FMNIST dataset, split it into training, validation and fitness sets, and preprocess when applicable.
        # FMNIST test set is not used in the evolution, it is only used in the final benchmark evaluation, so we can ignore it here.
        (x, y), (_, _) = fmnist.load_data()

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
        
        # Convert labels to one-hot encoding
        y = keras.utils.to_categorical(y, self.n_classes)

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

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_fit = x_fit
        self.y_fit = y_fit
    
    def load_data_for_benchmark(self):
        # Load the FMNIST dataset, split it into training, validation sets.
        # Load the test set of FMNIST, separately.
        # Preprocess when applicable.
        (x, y), (x_test, y_test) = fmnist.load_data()

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
        
        # Convert labels to one-hot encoding
        y = keras.utils.to_categorical(y, self.n_classes)
        y_test = keras.utils.to_categorical(y_test, self.n_classes)

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