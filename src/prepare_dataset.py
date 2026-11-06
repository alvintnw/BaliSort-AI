"""Prepare the raw BaliSort AI dataset for model training."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import TypeAlias

from PIL import Image, UnidentifiedImageError


CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
SPLITS: tuple[str, ...] = ("train", "validation", "test")
SplitImages: TypeAlias = dict[str, list[Path]]


def is_valid_image(path: Path) -> bool:
    """Return whether Pillow can verify the image without decoding it fully."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, UnidentifiedImageError):
        return False
    return True


def collect_images(
    source_dir: Path,
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Collect valid images by class and return invalid image paths separately."""
    images_by_class: dict[str, list[Path]] = {class_name: [] for class_name in CLASSES}
    corrupted_images: list[Path] = []

    for class_name in CLASSES:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if is_valid_image(path):
                images_by_class[class_name].append(path)
            else:
                corrupted_images.append(path)

    return images_by_class, corrupted_images


def split_images(
    images: list[Path],
    random_generator: random.Random,
) -> SplitImages:
    """Shuffle images and split them into 70%/15%/15% partitions."""
    shuffled_images = images.copy()
    random_generator.shuffle(shuffled_images)

    training_end = int(len(shuffled_images) * 0.70)
    validation_end = training_end + int(len(shuffled_images) * 0.15)
    return {
        "train": shuffled_images[:training_end],
        "validation": shuffled_images[training_end:validation_end],
        "test": shuffled_images[validation_end:],
    }


def create_output_directories(output_dir: Path) -> None:
    """Create the split and class directories required by the processed dataset."""
    for split in SPLITS:
        for class_name in CLASSES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_splits(
    split_images_by_class: dict[str, SplitImages],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Copy split images and return per-class counts for each split."""
    counts = {
        split: {class_name: 0 for class_name in CLASSES}
        for split in SPLITS
    }

    for class_name, split_images in split_images_by_class.items():
        for split, images in split_images.items():
            destination_dir = output_dir / split / class_name
            for source_path in images:
                try:
                    shutil.copy2(source_path, destination_dir / source_path.name)
                except OSError as error:
                    print(f"Gagal menyalin {source_path}: {error}")
                    continue
                counts[split][class_name] += 1

    return counts


def print_summary(
    counts: dict[str, dict[str, int]],
    corrupted_images: list[Path],
) -> None:
    """Print per-class counts, split totals, and corrupted image paths."""
    print("\nJumlah gambar hasil pemrosesan:")
    for split in SPLITS:
        split_counts = counts[split]
        total = sum(split_counts.values())
        details = ", ".join(
            f"{class_name}: {split_counts[class_name]}" for class_name in CLASSES
        )
        print(f"- {split}: {total} ({details})")

    print("\nDaftar gambar rusak:")
    if corrupted_images:
        for path in corrupted_images:
            print(f"- {path}")
    else:
        print("- Tidak ada")


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> int:
    """Validate, split, and copy the raw dataset into its processed layout."""
    if not source_dir.is_dir():
        print(f"Direktori sumber tidak ditemukan: {source_dir}")
        return 1

    images_by_class, corrupted_images = collect_images(source_dir)
    split_images_by_class = {
        class_name: split_images(images, random.Random(seed))
        for class_name, images in images_by_class.items()
    }

    create_output_directories(output_dir)
    counts = copy_splits(split_images_by_class, output_dir)
    print_summary(counts, corrupted_images)
    return 0


def main() -> int:
    """Run dataset preparation using paths relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    return prepare_dataset(
        source_dir=project_root / "dataset" / "raw" / "extracted" / "dataset-resized",
        output_dir=project_root / "dataset" / "processed",
    )


if __name__ == "__main__":
    raise SystemExit(main())
