from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam, SGD
from utils.data_functions import load_cifar10_full, load_cifar10_training, load_fashion_mnist_training
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class ResNet_Interface:
  # For some reason this model seems to want data without the standard /= 255 operation
  def __init__(self, incoming_data_shape=(28,28,1)):
    self.incoming_data_shape = incoming_data_shape
    self.model = None
    self.input_layer_shape = (224,224,3)
    #self.initialize_model()
    self.add_feature_extractor_to_model()
    #self.add_classifier_to_model()
  
  def get_model(self):
    return self.model

  def get_input_layer_shape(self):
    return self.input_layer_shape
  
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def add_feature_extractor_to_model(self):

    feature_extractor = ResNet50(input_shape=self.incoming_data_shape,
                                                include_top=True,
                                                weights=None, classes=200)
    self.model = feature_extractor
    self.model.trainable = True
    
    for layer in self.model.layers:
      print(f"Adding layer {layer.name} to model, with input shape {layer.input_shape} and output shape {layer.output_shape}")
      #self.model.add(layer)
    return

  def prepare_input(self, data):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
    print(f"Detected input shape {data.shape}")

    if len(data.shape) == 4 and data.shape[3] == 1:
      data = data[:,:,:,0]

    if len(data.shape) == 3:
      data = np.stack([data] * 3, axis=-1)

    data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in data])

    print(f"Changed to input shape {data.shape}")
    return preprocess_input(data)

