import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist, cifar10, mnist
from tensorflow.keras import backend as K
import numpy as np
from datasets import load_dataset

def resize_data(args):
    """
        Resize the dataset 28 x 28 datasets to 32x32

        Parameters
        ----------
        args : tuple(np.array, (int, int))
            instances, and shape of the reshaped signal

        Returns
        -------
        content : np.array
            reshaped instances
    """

    content, shape = args
    content = content.reshape(-1, 28, 28, 1)
    content = tf.convert_to_tensor(content)

    if shape != (28, 28):
        content = tf.image.resize(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    content = tf.image.grayscale_to_rgb(content)

    return content

def load_data_evolution(n_classes=10, validation_size=None, test_size=None, split=False, img_size = None, channels=1):
    img_rows, img_cols = img_size[0], img_size[1]
    (x_train, y_train), (_, _) = fashion_mnist.load_data()
    #x_train = resize_data((x_train, (img_rows, img_cols)))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                        test_size=validation_size + test_size,
                                                        stratify=y_train)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                        test_size=test_size,
                                                        stratify=y_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    if split:
        train_size = len(x_train)
        x_train_list = []
        y_train_list = []

        for i in range(6):
            x_train, foo, y_train, bar = train_test_split(x_train, y_train,
                                                            test_size=train_size // 7,
                                                            stratify=y_train)
            x_train_list.append(foo)
            y_train_list.append(bar)

        x_train_list.append(x_train)
        y_train_list.append(y_train)

        dataset = { 
            'x_train': x_train_list,
            'y_train': y_train_list,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}
    else:
        dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
                'y_test': y_test}
    return dataset

def load_cifar10_full(n_classes=10, validation_size=3500, test_size=3500):
    #Confirmar mnist
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)



    img_rows, img_cols, channels = 32, 32, 3

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_cifar10_training(n_classes=10, training_size=None, validation_size=None, test_size=None, normalize=True, subtract_mean=True):
    if training_size != None and validation_size != None and test_size != None:
        assert training_size + validation_size + test_size == 50000
    if test_size == None:
        assert training_size != None and validation_size != None
        test_size = 50000 - training_size - validation_size
    (x_train, y_train), (_, _) = cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=test_size,
                                                    stratify=y_val,
                                                    random_state=0)



    img_rows, img_cols, channels = 32, 32, 3

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    
    if normalize:    
        x_train /= 255
        x_val /= 255
        x_test /= 255

    #subraction of the mean image
    if subtract_mean:
        x_mean = 0
        for x in x_train:
            x_mean += x
        x_mean /= len(x_train)
        x_train -= x_mean
        x_val -= x_mean
        x_test -= x_mean

    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_fashion_mnist_full(n_classes=10, validation_size=3500):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size,
                                                    stratify=y_train,
                                                    random_state=0)



    img_rows, img_cols, channels = 28, 28, 1

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def reshape_fashion_mnist_training(x_train, y_train, n_classes=10, validation_size=3500, test_size=3500):

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=test_size,
                                                    stratify=y_val,
                                                    random_state=0)

    img_rows, img_cols, channels = 28, 28, 1

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def select_fashion_mnist_training(fashion, n_classes=10, validation_size=3500, test_size=3500):
    (x_train, y_train), (_, _) = fashion

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=test_size,
                                                    stratify=y_val,
                                                    random_state=0)

    img_rows, img_cols, channels = 28, 28, 1

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_fashion_mnist_training(n_classes=10, training_size=None, validation_size=None, test_size=None, normalize=True, subtract_mean=True):

    if training_size != None and validation_size != None and test_size != None:
        assert training_size + validation_size + test_size == 60000
    if test_size == None:
        assert training_size != None and validation_size != None
        test_size = 60000 - training_size - validation_size

    (x_train, y_train), (_, _) = fashion_mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=test_size,
                                                    stratify=y_val,
                                                    random_state=0)

    img_rows, img_cols, channels = 28, 28, 1


    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    if normalize:    
        x_train /= 255
        x_val /= 255
        x_test /= 255

    #subraction of the mean image
    if subtract_mean:
        x_mean = 0
        for x in x_train:
            x_mean += x
        x_mean /= len(x_train)
        x_train -= x_mean
        x_val -= x_mean
        x_test -= x_mean

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_mnist_full(n_classes=10, validation_size=3500, test_size=3500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)



    img_rows, img_cols, channels = 28, 28, 1

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_mnist_training(n_classes=10, validation_size=3500, test_size=3500):

    (x_train, y_train), (_, _) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train,
                                                    random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=test_size,
                                                    stratify=y_val,
                                                    random_state=0)

    img_rows, img_cols, channels = 28, 28, 1

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

def load_imagenet_training(n_classes=1000, validation_size=5000, test_size=0, batch_size=64, data_length=50000, normalize=True, subtract_mean=True):
    test_size = 0
    train_data = keras.preprocessing.image_dataset_from_directory(
        'imagenet/',
        labels='inferred',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=batch_size,
        validation_split=(validation_size + test_size) / data_length,
        subset='training',
        shuffle=True,
        seed=42)

    validation_data = keras.preprocessing.image_dataset_from_directory(
        'imagenet/',
        labels='inferred',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=batch_size,
        validation_split=(validation_size + test_size) / 50000,
        subset='validation',
        shuffle=True,
        seed=42)

    return train_data, validation_data

def load_tiny_imagenet(n_classes=200, validation_size=5000, test_size=0, batch_size=64, data_length=100000, normalize=True, subtract_mean=True):
    train_dataset = load_dataset("zh-plus/tiny-imagenet", split='train').with_format("numpy")
    validation_dataset = load_dataset("zh-plus/tiny-imagenet", split='valid').with_format("numpy")

    print(f"train len: {len(train_dataset)}, val len: {len(validation_dataset)}")
    y_train = keras.utils.to_categorical(train_dataset['label'], n_classes)
    y_val = keras.utils.to_categorical(validation_dataset['label'], n_classes)

    x_train = train_dataset['image']
    x_train = np.stack([x if x.shape == (64, 64, 3) else np.stack([x]*3, axis=-1) for x in x_train])
    y_train = keras.utils.to_categorical(train_dataset['label'], n_classes)

    x_val = validation_dataset['image']
    x_val = np.stack([x if x.shape == (64, 64, 3) else np.stack([x]*3, axis=-1) for x in x_val])


    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    img_rows, img_cols, channels = 64, 64, 3

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                        test_size=test_size,
                                                        stratify=y_val)
    dataset = { 
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        }
    return dataset