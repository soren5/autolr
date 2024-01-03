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

def adapt_mobile(input_shape=(32,32,3)):
  from tensorflow.keras.applications.mobilenet import MobileNet
  from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(model):

    feature_extractor = MobileNet(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')
    #(inputs)
    for layer in feature_extractor.layers:
       model.add(layer)
    return model


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(model):
      model.add(tf.keras.layers.GlobalAveragePooling2D())
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(1024, activation="relu"))
      model.add(tf.keras.layers.Dense(512, activation="relu"))
      model.add(tf.keras.layers.Dense(10, activation="softmax", name="classification"))
      return model

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(model, input_shape=(32,32,3)):
      size = int(224/input_shape[0])
      channels = int(3/input_shape[2])

      resize = tf.keras.layers.UpSampling3D(size=(size,size,channels))
      #if channels == 1:
      #  resize = tf.concat([resize] * 3, axis=-1)
      #elif channels == 3:
      #  pass
      #else:
      #  raise Exception("Invalid channel number")
      model.add(resize)
      model = feature_extractor(model)
      model = classifier(model)

      return model

  '''
  Define the model and compile it.
  '''
  def define_compile_model(input_shape=(32,32,3)):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    model = final_model(model, input_shape)
    
    return model
  
  def adjust_input(inputs):
    if inputs.shape[3] == 1:
      print("Fixing number of channels")
      inputs=np.dstack([inputs] * 3)
    elif inputs.shape[3] != 3:
      raise Exception("Number of channels in data is not 1 or 3.")
    return preprocess_input(inputs)

  return define_compile_model, adjust_input

def adapt_vgg16(input_shape=(32,32,3)):
  from tensorflow.keras.applications.vgg16 import VGG16
  from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(model):

    feature_extractor = VGG16(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')
    for layer in feature_extractor.layers:
       model.add(layer)
    return model


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(model):
      model.add(tf.keras.layers.GlobalAveragePooling2D())
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(1024, activation="relu"))
      model.add(tf.keras.layers.Dense(512, activation="relu"))
      model.add(tf.keras.layers.Dense(10, activation="softmax", name="classification"))
      return model

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(model, input_shape=(32,32,3)):
      size = int(224/input_shape[0])
      channels = int(3/input_shape[2])

      resize = tf.keras.layers.UpSampling3D(size=(size,size,channels))
      #if channels == 1:
      #  resize = tf.concat([resize] * 3, axis=-1)
      #elif channels == 3:
      #  pass
      #else:
      #  raise Exception("Invalid channel number")
      model.add(resize)
      model = feature_extractor(model)
      model = classifier(model)

      return model

  '''
  Define the model and compile it.
  '''
  def define_compile_model(input_shape=(32,32,3)):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    model = final_model(model, input_shape)
    
    return model
  
  def adjust_input(inputs):
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
    import matplotlib.pyplot as pplot 
    import imageio

    if inputs.shape[3] == 1:
      print(f"Detected input shape {inputs[0].shape}")
      #inputs = (inputs + 1) / 2

      #for x in range(10):
      #  img = Image.fromarray(inputs[x])
      #  img.save(f'image_samples/test{x}.png')

      inputs=np.concatenate([inputs] * 3, 3)
      print(f"Changed to input shape {inputs.shape}")
      for x in range(10):
        print(inputs[x].shape)
        imageio.imwrite(f'image_samples/sample{x}.jpg', inputs[x])
    elif inputs.shape[3] != 3:
      raise Exception("Number of channels in data is not 1 or 3.")
    return preprocess_input(inputs)
  return define_compile_model, adjust_input


def try_mobile_none():
  from tensorflow.keras.applications.mobilenet import MobileNet
  from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = MobileNet(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights=None)(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input


def try_xception():
  from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = Xception(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_vgg16():
  from tensorflow.keras.applications.vgg16 import VGG16 as net
  from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_resnet():
  from tensorflow.keras.applications.resnet50 import ResNet50 as net
  from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_inception():
  from tensorflow.keras.applications.inception_v3 import InceptionV3 as net
  from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_densenet():
  from tensorflow.keras.applications.densenet import DenseNet121 as net
  from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_nasnet():
  from tensorflow.keras.applications.nasnet import NASNetMobile as net
  from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input

def try_efficient():
  from tensorflow.keras.applications.efficientnet import EfficientNetB0 as net
  from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = net(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet')(inputs)
    return feature_extractor


  '''
  Defines final dense layers and subsequent softmax layer for classification.
  '''
  def classifier(inputs):
      x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(1024, activation="relu")(x)
      x = tf.keras.layers.Dense(512, activation="relu")(x)
      x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
      return x

  '''
  Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
  Connect the feature extraction and "classifier" layers to build the model.
  '''
  def final_model(inputs):

      resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

      resnet_feature_extractor = feature_extractor(resize)
      classification_output = classifier(resnet_feature_extractor)

      return classification_output

  '''
  Define the model and compile it. 
  Use Stochastic Gradient Descent as the optimizer.
  Use Sparse Categorical CrossEntropy as the loss function.
  '''
  def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    
    return model
  return define_compile_model, preprocess_input


def try_model(try_function):
  from tensorflow.keras.utils import to_categorical
  from sklearn.model_selection import train_test_split

  define_compile_model, preproc = try_function()
  model = define_compile_model((224,224,3))


  #dataset = load_fashion_mnist_training(training_size=44900, validation_size=100)
  #dataset['x_train'] = preproc(dataset['x_train'])
  #dataset['x_val'] = preproc(dataset['x_val'])
  #print(model.summary())
  from bayes_opt import BayesianOptimization

  #Despair
  from tensorflow.keras.datasets import fashion_mnist
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                              test_size=55000,
                              stratify=y_train)
  print(x_train.shape)
  x_train = np.stack([x_train] * 3, axis=-1)
  print(x_train.shape)
  from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
  x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in x_train])
  print(x_train.shape)

  y_train = to_categorical(y_train)

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
  bayesian_optimizer = BayesianOptimization(f=create_baye_opt(model, x_train, y_train), pbounds={'lr': (0,1)}, verbose=2)
  bayesian_optimizer.probe(params={'lr': 0.01})
  bayesian_optimizer.maximize(init_points=10, n_iter=1000)

if __name__ == "__main__":
  try_model(adapt_vgg16)
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