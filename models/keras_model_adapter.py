from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam, SGD
from utils.data_functions import load_cifar10_full, load_cifar10_training, load_fashion_mnist_training
import tensorflow as tf
from tensorflow.keras import Sequential
import cv2 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class MobileNet_Interface:
  def __init__(self, incoming_data_shape=(32,32,3)):
    self.incoming_data_shape = incoming_data_shape
    self.model = None
    self.input_layer_shape = (224,224,3)
    self.initialize_model()
    self.add_feature_extractor_to_model()
    self.add_classifier_to_model()

  def get_model(self):
    return self.model
  
  def get_input_layer_shape(self):
    return self.input_layer_shape
  
  def add_feature_extractor_to_model(self):
    from tensorflow.keras.applications.mobilenet import MobileNet

    feature_extractor = MobileNet(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')
    for layer in feature_extractor.layers:
       self.model.add(layer)


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def add_classifier_to_model(self):
      self.model.add(tf.keras.layers.GlobalAveragePooling2D())
      self.model.add(tf.keras.layers.Flatten())
      self.model.add(tf.keras.layers.Dense(1024, activation="relu"))
      self.model.add(tf.keras.layers.Dense(512, activation="relu"))
      self.model.add(tf.keras.layers.Dense(10, activation="softmax", name="classification"))

  '''
  Define the model and compile it.
  '''
  def initialize_model(self):
    self.model = Sequential()
    self.model.add(tf.keras.layers.Input(shape=self.input_layer_shape))
  
  def prepare_input(self, data):
    from tensorflow.keras.applications.mobilenet import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
    print(f"Detected input shape {data.shape}")

    if len(data.shape) == 4 and data.shape[3] == 1:
      data = data[:,:,:,0]

    if len(data.shape) == 3:
      data = np.stack([data] * 3, axis=-1)

    data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in data])

    print(f"Changed to input shape {data.shape}")
    return preprocess_input(data)

class VGG16_Interface:
  # For some reason this model seems to want data without the standard /= 255 operation
  def __init__(self, incoming_data_shape=(28,28,1)):
    self.incoming_data_shape = incoming_data_shape
    self.model = None
    self.input_layer_shape = (224,224,3)
    self.initialize_model()
    self.add_feature_extractor_to_model()
    self.add_classifier_to_model()
  
  def get_model(self):
    return self.model

  def get_input_layer_shape(self):
    return self.input_layer_shape
  
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def add_feature_extractor_to_model(self):
    from tensorflow.keras.applications.vgg16 import VGG16

    feature_extractor = VGG16(input_shape=self.input_layer_shape,
                                                include_top=False,
                                                weights='imagenet')
    for layer in feature_extractor.layers:
       self.model.add(layer)

  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def add_classifier_to_model(self):
      self.model.add(tf.keras.layers.GlobalAveragePooling2D())
      self.model.add(tf.keras.layers.Flatten())
      self.model.add(tf.keras.layers.Dense(1024, activation="relu"))
      self.model.add(tf.keras.layers.Dense(512, activation="relu"))
      self.model.add(tf.keras.layers.Dense(10, activation="softmax", name="classification"))

  def initialize_model(self):
    self.model = Sequential()
    self.model.add(tf.keras.layers.Input(shape=self.input_layer_shape))
  
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


def try_model():
  from tensorflow.keras.utils import to_categorical
  from sklearn.model_selection import train_test_split    
  from tensorflow.keras.applications.vgg16 import preprocess_input as preproc


  vgg16 = VGG16_Interface()
  model = vgg16.get_model()

  #Despair
  from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_mnist_training, select_fashion_mnist_training
  from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
  data = load_fashion_mnist_training(training_size=5000, validation_size=100, normalize=False, subtract_mean=False)
  x_train = data['x_train'][:,:,:,0]
  y_train = data['y_train']
  #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
  #                            test_size=55000,
  #                            stratify=y_train)
  print(x_train.shape)
  x_train = np.stack([x_train] * 3, axis=-1)
  print(x_train.shape)
  x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in x_train])
  print(x_train.shape)

  #y_train = to_categorical(y_train)

  x_train = preproc(x_train)
  model.save_weights('models/weights')
  def create_baye_opt(model, x_train, y_train):
    def baye_opt(lr):
      model.load_weights('models/weights')
      optimizer = SGD(lr)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
      score = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          verbose=2,
          )
      return max(score.history['accuracy'])
    return baye_opt
  for x in range(30):
    create_baye_opt(model, x_train, y_train)(0.01)

if __name__ == "__main__":
  try_model()
"""
#try_model(try_mobile_none)
try_model(try_xception)
try_model(try_vgg16)
try_model(try_resnet)
try_model(try_inception)
try_model(try_densenet)
try_model(try_nasnet)
try_model(try_efficient)
"""