import numpy as np


def _build_canonical_tiny_imagenet_tree(root, class_count=3, train_per_class=4, val_per_class=1):
    dataset_root = root / "tiny_imagenet"
    class_names = [f"n{i:08d}" for i in range(class_count)]
    (dataset_root / "val" / "images").mkdir(parents=True)
    (dataset_root / "wnids.txt").write_text("\n".join(class_names) + "\n")

    annotations = []
    for class_name in class_names:
        class_images = dataset_root / "train" / class_name / "images"
        class_images.mkdir(parents=True)
        for image_index in range(train_per_class):
            (class_images / f"{class_name}_train_{image_index}.JPEG").write_text("")
        for image_index in range(val_per_class):
            image_name = f"{class_name}_val_{image_index}.JPEG"
            (dataset_root / "val" / "images" / image_name).write_text("")
            annotations.append(f"{image_name}\t{class_name}\t0\t0\t1\t1\n")
    (dataset_root / "val" / "val_annotations.txt").write_text("".join(annotations))
    return dataset_root, class_names


def test_canonical_tiny_imagenet_loader_reads_stanford_layout(tmp_path, monkeypatch):
    import dataset_loaders.tiny_imagenet as tiny_imagenet
    from dataset_loaders.tiny_imagenet import TINY_IMAGENET_Dataset

    dataset_root = tmp_path / "tiny_imagenet"
    (dataset_root / "train" / "n00000001" / "images").mkdir(parents=True)
    (dataset_root / "train" / "n00000002" / "images").mkdir(parents=True)
    (dataset_root / "val" / "images").mkdir(parents=True)
    (dataset_root / "wnids.txt").write_text("n00000001\nn00000002\n")
    (dataset_root / "val" / "val_annotations.txt").write_text(
        "val_0.JPEG\tn00000002\t0\t0\t1\t1\n"
        "val_1.JPEG\tn00000001\t0\t0\t1\t1\n"
    )
    (dataset_root / "train" / "n00000001" / "images" / "train_0.JPEG").write_text("")
    (dataset_root / "train" / "n00000002" / "images" / "train_1.JPEG").write_text("")
    (dataset_root / "val" / "images" / "val_0.JPEG").write_text("")
    (dataset_root / "val" / "images" / "val_1.JPEG").write_text("")

    def fake_imread(path):
        path = str(path)
        if path.endswith("train_0.JPEG"):
            return np.zeros((64, 64), dtype=np.uint8)
        if path.endswith("val_0.JPEG"):
            return np.zeros((64, 64, 4), dtype=np.uint8)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(tiny_imagenet.plt, "imread", fake_imread)

    (x, y), (x_test, y_test) = TINY_IMAGENET_Dataset(path=str(dataset_root)).load_data()

    assert x.shape == (2, 64, 64, 3)
    assert x_test.shape == (2, 64, 64, 3)
    assert y.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert y_test.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_custom_tiny_imagenet_loader_defaults_to_custom_folder(tmp_path):
    from dataset_loaders.tiny_imagenet_custom import TINY_IMAGENET_CUSTOM_Dataset
    from sge.parameters import params

    params["DATA_DIR"] = str(tmp_path)

    dataset = TINY_IMAGENET_CUSTOM_Dataset()

    assert dataset.path == str(tmp_path / "tiny_imagenet_custom")


def test_canonical_tiny_imagenet_evolution_splits_paths_before_loading(tmp_path, monkeypatch):
    import dataset_loaders.tiny_imagenet as tiny_imagenet
    from dataset_loaders.tiny_imagenet import TINY_IMAGENET_Dataset

    dataset_root, _ = _build_canonical_tiny_imagenet_tree(tmp_path)
    loaded_paths = []

    def fake_imread(path):
        loaded_paths.append(str(path))
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(tiny_imagenet.plt, "imread", fake_imread)
    dataset = TINY_IMAGENET_Dataset(
        validation_size=3,
        fitness_size=3,
        seed=1,
        normalize=False,
        subtract_mean=False,
        path=str(dataset_root),
    )
    dataset.load_data = lambda: (_ for _ in ()).throw(AssertionError("load_data used"))

    dataset.load_data_for_evolution()

    assert dataset.x_train.shape == (6, 64, 64, 3)
    assert dataset.x_val.shape == (3, 64, 64, 3)
    assert dataset.x_fit.shape == (3, 64, 64, 3)
    assert dataset.y_train.shape == (6, 3)
    assert dataset.y_val.shape == (3, 3)
    assert dataset.y_fit.shape == (3, 3)
    assert len(loaded_paths) == 12
    assert all("/train/" in path for path in loaded_paths)


def test_canonical_tiny_imagenet_benchmark_splits_paths_before_loading(tmp_path, monkeypatch):
    import dataset_loaders.tiny_imagenet as tiny_imagenet
    from dataset_loaders.tiny_imagenet import TINY_IMAGENET_Dataset

    dataset_root, _ = _build_canonical_tiny_imagenet_tree(tmp_path)
    loaded_paths = []

    def fake_imread(path):
        loaded_paths.append(str(path))
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(tiny_imagenet.plt, "imread", fake_imread)
    dataset = TINY_IMAGENET_Dataset(
        validation_size=3,
        fitness_size=3,
        seed=1,
        normalize=False,
        subtract_mean=False,
        path=str(dataset_root),
    )
    dataset.test_size = 3
    dataset.load_data = lambda: (_ for _ in ()).throw(AssertionError("load_data used"))

    dataset.load_data_for_benchmark()

    assert dataset.x_train.shape == (9, 64, 64, 3)
    assert dataset.x_val.shape == (3, 64, 64, 3)
    assert dataset.x_test.shape == (3, 64, 64, 3)
    assert len(loaded_paths) == 15
    assert sum("/train/" in path for path in loaded_paths) == 12
    assert sum("/val/images/" in path for path in loaded_paths) == 3


def test_canonical_tiny_imagenet_subtracts_train_split_mean(tmp_path, monkeypatch):
    import dataset_loaders.tiny_imagenet as tiny_imagenet
    from dataset_loaders.tiny_imagenet import TINY_IMAGENET_Dataset

    dataset_root, class_names = _build_canonical_tiny_imagenet_tree(tmp_path)
    class_values = {class_name: index * 10 for index, class_name in enumerate(class_names)}

    def fake_imread(path):
        path = str(path)
        class_name = next(name for name in class_names if name in path)
        return np.full((64, 64, 3), class_values[class_name], dtype=np.uint8)

    monkeypatch.setattr(tiny_imagenet.plt, "imread", fake_imread)
    dataset = TINY_IMAGENET_Dataset(
        validation_size=3,
        fitness_size=3,
        seed=1,
        normalize=False,
        subtract_mean=True,
        path=str(dataset_root),
    )

    dataset.load_data_for_evolution()

    assert dataset.x_train.mean() == np.float32(0.0)
    assert set(np.unique(dataset.x_train).tolist()) == {-10.0, 0.0, 10.0}
    assert set(np.unique(dataset.x_val).tolist()) <= {-10.0, 0.0, 10.0}
    assert set(np.unique(dataset.x_fit).tolist()) <= {-10.0, 0.0, 10.0}
