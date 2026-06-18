import json
import zipfile
from pathlib import Path

import pytest


def _write_fake_canonical_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tiny-imagenet-200/wnids.txt", "n00000001\nn00000002\n")
        archive.writestr(
            "tiny-imagenet-200/words.txt",
            "n00000001\tfirst class\nn00000002\tsecond class\n",
        )
        for wnid in ["n00000001", "n00000002"]:
            for index in range(2):
                archive.writestr(
                    f"tiny-imagenet-200/train/{wnid}/images/{wnid}_{index}.JPEG",
                    f"{wnid}-{index}",
                )
        archive.writestr(
            "tiny-imagenet-200/val/val_annotations.txt",
            "val_0.JPEG\tn00000001\t0\t0\t1\t1\n"
            "val_1.JPEG\tn00000002\t0\t0\t1\t1\n",
        )
        archive.writestr("tiny-imagenet-200/val/images/val_0.JPEG", "val-0")
        archive.writestr("tiny-imagenet-200/val/images/val_1.JPEG", "val-1")
        archive.writestr("tiny-imagenet-200/test/images/test_0.JPEG", "test-0")


def test_install_from_archive_converts_to_loader_layout(tmp_path):
    from utils.download_tiny_imagenet import install_from_archive

    archive_path = tmp_path / "tiny-imagenet-200.zip"
    output_dir = tmp_path / "data" / "tiny_imagenet"
    _write_fake_canonical_archive(archive_path)

    manifest = install_from_archive(
        archive_path,
        output_dir,
        source_url="https://example.test/tiny-imagenet-200.zip",
    )

    assert manifest["classes"] == 2
    assert manifest["train_images"] == 4
    assert manifest["val_images"] == 2
    assert manifest["test_images"] == 1
    assert (output_dir / "train" / "n00000001" / "n00000001_0.JPEG").read_text() == "n00000001-0"
    assert (output_dir / "val" / "n00000002" / "val_1.JPEG").read_text() == "val-1"
    assert (output_dir / "test" / "images" / "test_0.JPEG").read_text() == "test-0"
    assert (output_dir / "metadata" / "wnids.txt").is_file()
    saved_manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
    assert saved_manifest["layout"] == "autolr_loader"


def test_install_from_archive_refuses_to_replace_existing_output(tmp_path):
    from utils.download_tiny_imagenet import install_from_archive

    archive_path = tmp_path / "tiny-imagenet-200.zip"
    output_dir = tmp_path / "data" / "tiny_imagenet"
    _write_fake_canonical_archive(archive_path)
    output_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        install_from_archive(archive_path, output_dir)


def test_parse_val_annotations(tmp_path):
    from utils.download_tiny_imagenet import parse_val_annotations

    annotations_path = tmp_path / "val_annotations.txt"
    annotations_path.write_text(
        "val_0.JPEG\tn00000001\t0\t0\t1\t1\n"
        "val_1.JPEG\tn00000002\t0\t0\t1\t1\n"
    )

    assert parse_val_annotations(annotations_path) == {
        "val_0.JPEG": "n00000001",
        "val_1.JPEG": "n00000002",
    }
