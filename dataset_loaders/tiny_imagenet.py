from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_loaders.dataset_utils import validate_benchmark_test_size


class TINY_IMAGENET_Dataset:
    def __init__(self, validation_size=3500, fitness_size=3500, seed=0, normalize=True, subtract_mean=True, path=None):
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
        (x, y), (_, _) = self.load_data()

        x = x.astype('float32')
        if self.normalize:
            x /= 255

        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], self.channels, self.img_rows, self.img_cols)
        else:
            x = x.reshape(x.shape[0], self.img_rows, self.img_cols, self.channels)

        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                        test_size=self.validation_size + self.fitness_size,
                                                        stratify=y,
                                                        random_state=self.seed)
        x_val, x_fit, y_val, y_fit = train_test_split(x_val, y_val,
                                                        test_size=self.fitness_size,
                                                        stratify=y_val,
                                                        random_state=self.seed)

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
        (x, y), (x_test, y_test) = self.load_data()

        x = x.astype('float32')
        x_test = x_test.astype('float32')
        if self.normalize:
            x /= 255
            x_test /= 255

        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], self.channels, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], self.channels, self.img_rows, self.img_cols)
        else:
            x = x.reshape(x.shape[0], self.img_rows, self.img_cols, self.channels)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.channels)

        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                        test_size=self.validation_size,
                                                        stratify=y,
                                                        random_state=self.seed)

        if self.subtract_mean:
            x_mean = 0
            for x in x_train:
                x_mean += x
            x_mean /= len(x_train)
            x_train -= x_mean
            x_val -= x_mean
            x_test -= x_mean

        validate_benchmark_test_size(
            x_test, y_test, expected_test_size=getattr(self, "test_size", None)
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def load_data(self):
        class_names = self._load_class_names()
        x, y = self._load_training_data(class_names)
        x_test, y_test = self._load_validation_data(class_names)
        return (x, y), (x_test, y_test)

    def _load_class_names(self):
        metadata_wnids = os.path.join(self.path, 'metadata', 'wnids.txt')
        root_wnids = os.path.join(self.path, 'wnids.txt')
        if os.path.isfile(metadata_wnids):
            wnids_path = metadata_wnids
        elif os.path.isfile(root_wnids):
            wnids_path = root_wnids
        else:
            train_path = os.path.join(self.path, 'train')
            return sorted(
                d for d in os.listdir(train_path)
                if os.path.isdir(os.path.join(train_path, d))
            )

        with open(wnids_path) as wnids_file:
            return [line.strip() for line in wnids_file if line.strip()]

    def _one_hot(self, class_name, class_index):
        y_vector = np.zeros(len(class_index))
        y_vector[class_index[class_name]] = 1
        return y_vector

    def _load_training_data(self, class_names):
        train_path = os.path.join(self.path, 'train')
        class_index = {class_name: i for i, class_name in enumerate(class_names)}
        x = []
        y = []

        for class_name in class_names:
            class_path = os.path.join(train_path, class_name)
            images_path = os.path.join(class_path, 'images')
            if os.path.isdir(images_path):
                class_path = images_path
            for image_file in sorted(os.listdir(class_path)):
                if not image_file.endswith('.JPEG'):
                    continue
                image = plt.imread(os.path.join(class_path, image_file))
                x.append(image)
                y.append(self._one_hot(class_name, class_index))

        return np.array(x), np.array(y)

    def _load_validation_data(self, class_names):
        val_path = os.path.join(self.path, 'val')
        canonical_images_path = os.path.join(val_path, 'images')
        class_index = {class_name: i for i, class_name in enumerate(class_names)}
        x = []
        y = []

        if os.path.isdir(canonical_images_path):
            annotations = self._load_val_annotations()
            for image_file in sorted(os.listdir(canonical_images_path)):
                if not image_file.endswith('.JPEG'):
                    continue
                class_name = annotations[image_file]
                image = plt.imread(os.path.join(canonical_images_path, image_file))
                x.append(image)
                y.append(self._one_hot(class_name, class_index))
        else:
            for class_name in class_names:
                class_path = os.path.join(val_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for image_file in sorted(os.listdir(class_path)):
                    if not image_file.endswith('.JPEG'):
                        continue
                    image = plt.imread(os.path.join(class_path, image_file))
                    x.append(image)
                    y.append(self._one_hot(class_name, class_index))

        return np.array(x), np.array(y)

    def _load_val_annotations(self):
        annotation_paths = [
            os.path.join(self.path, 'metadata', 'val_annotations.txt'),
            os.path.join(self.path, 'val', 'val_annotations.txt'),
        ]
        for annotation_path in annotation_paths:
            if os.path.isfile(annotation_path):
                annotations = {}
                with open(annotation_path) as annotations_file:
                    for line in annotations_file:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            annotations[parts[0]] = parts[1]
                return annotations
        raise FileNotFoundError(
            f"Could not find Tiny ImageNet validation annotations under {self.path}"
        )
