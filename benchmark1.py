
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from utils.data_functions import load_cifar10_full
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def try_mobile():
  from tensorflow.keras.applications.mobilenet import MobileNet
  from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
  '''
  Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
  Input size is 224 x 224.
  '''
  def feature_extractor(inputs):

    feature_extractor = MobileNet(input_shape=(224, 224, 3),
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
  define_compile_model, preprocess_input = try_function()
  model = define_compile_model()

  optimizer = Adam()
  dataset = load_cifar10_full()
  print(model.summary())
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

  score = model.fit(preprocess_input(dataset['x_train']), dataset['y_train'],
      batch_size=32,
      epochs=1000,
      verbose=2,
      validation_data=(preprocess_input(dataset['x_val']), dataset['y_val']),
      validation_steps= len(dataset['x_val']) // 32,
      callbacks=[early_stop]
      )

#try_model(try_mobile)
try_model(try_mobile_none)
#try_model(try_xception)
#try_model(try_vgg16)
#try_model(try_resnet)
#try_model(try_inception)
#try_model(try_densenet)
#try_model(try_nasnet)
#try_model(try_efficient)