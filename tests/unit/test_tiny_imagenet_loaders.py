import numpy as np


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
