from pathlib import Path

import pytest

from dataset_loaders.dataset_utils import validate_benchmark_test_size


BENCHMARK_DATASET_LOADERS = [
    "dataset_loaders/fmnist.py",
    "dataset_loaders/mnist.py",
    "dataset_loaders/cifar10.py",
    "dataset_loaders/cifar100.py",
    "dataset_loaders/tiny_imagenet.py",
    "dataset_loaders/imagenet_100.py",
]


def test_validate_benchmark_test_size_accepts_matching_split():
    validate_benchmark_test_size([1, 2], [3, 4], expected_test_size=2)


def test_validate_benchmark_test_size_uses_framework_parameter_by_default():
    from sge.parameters import params

    params["TEST_SIZE"] = 2

    validate_benchmark_test_size([1, 2], [3, 4])


def test_validate_benchmark_test_size_requires_parameter():
    from sge.parameters import params

    assert "TEST_SIZE" not in params

    with pytest.raises(ValueError, match="TEST_SIZE must be provided"):
        validate_benchmark_test_size([1, 2], [3, 4])


def test_validate_benchmark_test_size_rejects_feature_label_mismatch():
    with pytest.raises(ValueError, match="x_test/y_test sizes differ"):
        validate_benchmark_test_size([1, 2], [3], expected_test_size=2)


def test_validate_benchmark_test_size_rejects_parameter_mismatch():
    with pytest.raises(ValueError, match="TEST_SIZE=3"):
        validate_benchmark_test_size([1, 2], [3, 4], expected_test_size=3)


@pytest.mark.parametrize("loader_path", BENCHMARK_DATASET_LOADERS)
def test_all_benchmark_dataset_loaders_validate_test_size(loader_path):
    source = Path(loader_path).read_text()
    benchmark_function = source.split("def load_data_for_benchmark", 1)[1]
    benchmark_function = benchmark_function.split("\n    def ", 1)[0]

    assert "validate_benchmark_test_size(" in benchmark_function
