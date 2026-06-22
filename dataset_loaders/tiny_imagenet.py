from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_loaders.dataset_utils import validate_benchmark_test_size


class TINY_IMAGENET_Dataset:
    """Canonical Tiny ImageNet loader with path-first splitting.

    Unlike the smaller Keras-backed loaders, canonical Tiny ImageNet is large
    enough that loading every image and then splitting arrays can OOM modest
    machines. The evaluator-facing API still exposes NumPy arrays, but this
    loader indexes image paths first, splits those lightweight records, and only
    materializes the final train/validation/fitness or test arrays.
    """

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
        class_names = self._load_class_names()
        class_index = self._class_index(class_names)
        train_examples = self._index_training_examples(class_names)

        train_examples, heldout_examples = train_test_split(
            train_examples,
            test_size=self.validation_size + self.fitness_size,
            stratify=self._example_class_names(train_examples),
            random_state=self.seed,
        )
        val_examples, fit_examples = train_test_split(
            heldout_examples,
            test_size=self.fitness_size,
            stratify=self._example_class_names(heldout_examples),
            random_state=self.seed,
        )

        x_train, y_train = self._load_examples(train_examples, class_index)
        x_val, y_val = self._load_examples(val_examples, class_index)
        x_fit, y_fit = self._load_examples(fit_examples, class_index)
        x_train, x_val, x_fit = self._preprocess_splits(x_train, x_val, x_fit)

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_fit = x_fit
        self.y_fit = y_fit

    def load_data_for_benchmark(self):
        class_names = self._load_class_names()
        class_index = self._class_index(class_names)
        train_examples = self._index_training_examples(class_names)
        test_examples = self._index_validation_examples(class_names)

        train_examples, val_examples = train_test_split(
            train_examples,
            test_size=self.validation_size,
            stratify=self._example_class_names(train_examples),
            random_state=self.seed,
        )

        x_train, y_train = self._load_examples(train_examples, class_index)
        x_val, y_val = self._load_examples(val_examples, class_index)
        x_test, y_test = self._load_examples(test_examples, class_index)
        x_train, x_val, x_test = self._preprocess_splits(x_train, x_val, x_test)

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
        class_index = self._class_index(class_names)
        x, y = self._load_examples(
            self._index_training_examples(class_names),
            class_index,
            preprocess=False,
        )
        x_test, y_test = self._load_examples(
            self._index_validation_examples(class_names),
            class_index,
            preprocess=False,
        )
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

    def _class_index(self, class_names):
        return {class_name: i for i, class_name in enumerate(class_names)}

    def _example_class_names(self, examples):
        return [class_name for _, class_name in examples]

    def _load_image(self, image_path):
        image = plt.imread(image_path)
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape != (self.img_rows, self.img_cols, self.channels):
            raise ValueError(
                f"Expected Tiny ImageNet image {image_path} to have shape "
                f"({self.img_rows}, {self.img_cols}, {self.channels}), got {image.shape}"
            )
        return image

    def _index_training_examples(self, class_names):
        train_path = os.path.join(self.path, 'train')
        examples = []

        for class_name in class_names:
            class_path = os.path.join(train_path, class_name)
            images_path = os.path.join(class_path, 'images')
            if os.path.isdir(images_path):
                class_path = images_path
            for image_file in sorted(os.listdir(class_path)):
                if not image_file.endswith('.JPEG'):
                    continue
                examples.append((os.path.join(class_path, image_file), class_name))

        return examples

    def _index_validation_examples(self, class_names):
        val_path = os.path.join(self.path, 'val')
        canonical_images_path = os.path.join(val_path, 'images')
        examples = []

        if os.path.isdir(canonical_images_path):
            annotations = self._load_val_annotations()
            for image_file in sorted(os.listdir(canonical_images_path)):
                if not image_file.endswith('.JPEG'):
                    continue
                class_name = annotations[image_file]
                examples.append((os.path.join(canonical_images_path, image_file), class_name))
        else:
            for class_name in class_names:
                class_path = os.path.join(val_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for image_file in sorted(os.listdir(class_path)):
                    if not image_file.endswith('.JPEG'):
                        continue
                    examples.append((os.path.join(class_path, image_file), class_name))

        return examples

    def _load_examples(self, examples, class_index, preprocess=True):
        if not preprocess:
            x = []
            y = np.zeros((len(examples), len(class_index)), dtype=np.float32)
            for i, (image_path, class_name) in enumerate(examples):
                x.append(self._load_image(image_path))
                y[i] = self._one_hot(class_name, class_index)
            return np.array(x), y

        x = np.empty(
            (len(examples), self.img_rows, self.img_cols, self.channels),
            dtype=np.float32,
        )
        y = np.zeros((len(examples), len(class_index)), dtype=np.float32)

        for i, (image_path, class_name) in enumerate(examples):
            x[i] = self._load_image(image_path)
            y[i] = self._one_hot(class_name, class_index)

        if preprocess:
            if self.normalize:
                x /= 255
            x = self._format_images(x)
        return x, y

    def _format_images(self, x):
        if K.image_data_format() == 'channels_first':
            return x.reshape(x.shape[0], self.channels, self.img_rows, self.img_cols)
        return x.reshape(x.shape[0], self.img_rows, self.img_cols, self.channels)

    def _preprocess_splits(self, x_train, *other_splits):
        if self.subtract_mean:
            x_mean = x_train.mean(axis=0)
            x_train -= x_mean
            for split in other_splits:
                split -= x_mean
        return (x_train,) + other_splits

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
