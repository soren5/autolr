"""Download and install the canonical Tiny ImageNet-200 dataset.

The Stanford archive uses a slightly different layout from AutoLR's current
``TINY_IMAGENET_Dataset`` loader. This utility downloads the canonical archive
and installs a loader-compatible copy:

    tiny_imagenet/train/<wnid>/*.JPEG
    tiny_imagenet/val/<wnid>/*.JPEG

The original metadata files are copied under ``tiny_imagenet/metadata``.
"""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
CANONICAL_ROOT = "tiny-imagenet-200"


def parse_val_annotations(path):
    """Return ``image filename -> wnid`` from canonical val annotations."""

    annotations = {}
    with Path(path).open() as annotations_file:
        for line in annotations_file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            annotations[parts[0]] = parts[1]
    return annotations


def download_archive(url, destination, redownload=False):
    """Download ``url`` to ``destination`` unless it already exists."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not redownload:
        return destination

    temporary_path = destination.with_name(f".{destination.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)
    return destination


def _canonical_root(extracted_dir):
    extracted_dir = Path(extracted_dir)
    root = extracted_dir / CANONICAL_ROOT
    if root.is_dir():
        return root

    candidates = [
        path
        for path in extracted_dir.iterdir()
        if path.is_dir() and (path / "wnids.txt").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not find canonical Tiny ImageNet root under {extracted_dir}"
    )


def _copy_metadata(source_root, destination_root):
    metadata_dir = destination_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in [
        "wnids.txt",
        "words.txt",
        "val/val_annotations.txt",
    ]:
        source = source_root / relative_path
        if source.is_file():
            shutil.copy2(source, metadata_dir / Path(relative_path).name)


def _copy_train_split(source_root, destination_root):
    train_output = destination_root / "train"
    train_output.mkdir(parents=True, exist_ok=True)
    train_count = 0
    class_count = 0

    for class_dir in sorted((source_root / "train").iterdir()):
        images_dir = class_dir / "images"
        if not class_dir.is_dir() or not images_dir.is_dir():
            continue
        output_class_dir = train_output / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        class_count += 1
        for image_path in sorted(images_dir.glob("*.JPEG")):
            shutil.copy2(image_path, output_class_dir / image_path.name)
            train_count += 1

    return class_count, train_count


def _copy_val_split(source_root, destination_root):
    val_annotations = parse_val_annotations(source_root / "val" / "val_annotations.txt")
    val_images_dir = source_root / "val" / "images"
    val_output = destination_root / "val"
    val_output.mkdir(parents=True, exist_ok=True)
    val_count = 0

    for image_path in sorted(val_images_dir.glob("*.JPEG")):
        wnid = val_annotations.get(image_path.name)
        if wnid is None:
            raise ValueError(f"No validation annotation found for {image_path.name}")
        output_class_dir = val_output / wnid
        output_class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_class_dir / image_path.name)
        val_count += 1

    return val_count


def _copy_test_split(source_root, destination_root):
    test_images_dir = source_root / "test" / "images"
    if not test_images_dir.is_dir():
        return 0

    test_output = destination_root / "test" / "images"
    test_output.mkdir(parents=True, exist_ok=True)
    test_count = 0
    for image_path in sorted(test_images_dir.glob("*.JPEG")):
        shutil.copy2(image_path, test_output / image_path.name)
        test_count += 1
    return test_count


def install_from_archive(archive_path, output_dir, force=False, source_url=None):
    """Install a loader-compatible Tiny ImageNet tree from a canonical zip."""

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"{output_dir} already exists. Use --force to replace it."
        )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-install-", dir=output_parent
    ) as temporary_root:
        temporary_root = Path(temporary_root)
        extracted_dir = temporary_root / "extracted"
        install_dir = temporary_root / output_dir.name
        extracted_dir.mkdir()
        install_dir.mkdir()

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted_dir)

        source_root = _canonical_root(extracted_dir)
        class_count, train_count = _copy_train_split(source_root, install_dir)
        val_count = _copy_val_split(source_root, install_dir)
        test_count = _copy_test_split(source_root, install_dir)
        _copy_metadata(source_root, install_dir)

        manifest = {
            "source_url": source_url,
            "archive": str(archive_path),
            "canonical_root": source_root.name,
            "layout": "autolr_loader",
            "classes": class_count,
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
        }
        with (install_dir / "dataset_manifest.json").open("w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(install_dir), str(output_dir))

    return manifest


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download canonical Tiny ImageNet-200 and install it in AutoLR's "
            "loader-compatible layout."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "data"),
        help="Base data directory; defaults to DATA_DIR or ./data.",
    )
    parser.add_argument(
        "--dataset-name",
        default="tiny_imagenet",
        help="Installed dataset folder name under --data-dir.",
    )
    parser.add_argument(
        "--archive",
        help="Use an existing tiny-imagenet-200.zip instead of downloading.",
    )
    parser.add_argument(
        "--download-dir",
        help="Where to store the downloaded zip; defaults to <data-dir>/downloads.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output.")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Download again even if the archive already exists.",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    data_dir = Path(args.data_dir)
    output_dir = data_dir / args.dataset_name

    if args.archive:
        archive_path = Path(args.archive)
    else:
        download_dir = Path(args.download_dir) if args.download_dir else data_dir / "downloads"
        archive_path = download_archive(
            args.url,
            download_dir / "tiny-imagenet-200.zip",
            redownload=args.redownload,
        )

    manifest = install_from_archive(
        archive_path,
        output_dir,
        force=args.force,
        source_url=args.url if not args.archive else None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Installed Tiny ImageNet at {output_dir}")


if __name__ == "__main__":
    main()
