from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam, SGD
from utils.data_functions import load_cifar10_full, load_cifar10_training, load_fashion_mnist_training
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from keras.engine import data_adapter
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def test_step(self, data):
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compute_loss(x, y, y_pred, sample_weight)
    print(f" compute loss: {self.compute_loss(x, y, y_pred, sample_weight)}")
    return self.compute_metrics(x, y, y_pred, sample_weight)

class ResNet_Interface:
  def __init__(self, incoming_data_shape=(28,28,1)):
    self.incoming_data_shape = incoming_data_shape
    self.model = ResNet50(input_shape=self.incoming_data_shape,
                                                include_top=True,
                                                weights=None, classes=200)
    self.model.trainable = True
    self.model.test_step = test_step
    self.pre_process = preprocess_input
  
  def get_model(self):
    return self.model
