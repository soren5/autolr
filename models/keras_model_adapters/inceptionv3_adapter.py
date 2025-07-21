from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam, SGD
from utils.data_functions import load_cifar10_full, load_cifar10_training, load_fashion_mnist_training
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class InceptionV3_Interface:
  # For some reason this model seems to want data without the standard /= 255 operation
  def __init__(self, incoming_data_shape=(28,28,1)):
    self.incoming_data_shape = incoming_data_shape
    self.model = InceptionV3(input_shape=self.incoming_data_shape,
                                                include_top=True,
                                                weights=None, classes=200)
    self.model.trainable = True
    self.pre_process = preprocess_input
  
  def get_model(self):
    return self.model

