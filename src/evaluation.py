"""Evaluate the best BaliSort AI MobileNetV3 checkpoint on the test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from src.model import CLASS_NAMES, build_model
    from src.preprocessing import get_eval_transforms
except ImportError:
    from model import CLASS_NAMES, build_model
    from preprocessing import get_eval_transforms


BATCH_SIZE = 32


def get_device() -> torch.device:
    """Return CUDA when available and otherwise use CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Menggunakan GPU: {torch.cuda.get_device_name(device)}")
        return device

    print("CUDA tidak tersedia; evaluasi dilanjutkan menggunakan CPU.")
    return torch.device("cpu")


def load_test_dataset(dataset_root: Path) -> ImageFolder:
    """Load the test split and verify its class-to-index mapping."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Direktori testing tidak ditemukan: {dataset_root}")

    dataset = ImageFolder(dataset_root, transform=get_eval_transforms())
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Mapping kelas dataset testing tidak sesuai. "
            f"Diharapkan {expected_mapping}, ditemukan {dataset.class_to_idx}."
        )
    if not dataset.samples:
        raise ValueError(f"Dataset testing kosong: {dataset_root}")
    return dataset


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load and validate checkpoint metadata without modifying the checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except (OSError, RuntimeError, EOFError) as error:
        raise RuntimeError(f"Gagal memuat checkpoint: {checkpoint_path}") from error

    if not isinstance(checkpoint, dict):
        raise ValueError("Format checkpoint tidak valid: harus berupa dictionary.")

    expected_classes = list(CLASS_NAMES)
    if checkpoint.get("class_names") != expected_classes:
        raise ValueError(
            "Urutan kelas checkpoint tidak sesuai. "
            f"Diharapkan {expected_classes}, ditemukan {checkpoint.get('class_names')}."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint tidak memiliki model_state_dict.")
    return checkpoint


def load_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build MobileNetV3 and load weights stored in the checkpoint."""
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError) as error:
        raise RuntimeError("Bobot checkpoint tidak cocok dengan MobileNetV3.") from error
    return model.to(device)


def run_inference(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference on the test loader and return targets and predictions."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            batch_predictions = outputs.argmax(dim=1)
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.cpu().tolist())
    return targets, predictions


def calculate_metrics(
    targets: list[int],
    predictions: list[int],
) -> tuple[float, float, float, float, str, Any]:
    """Calculate scalar metrics, a text report, and the confusion matrix."""
    labels = list(range(len(CLASS_NAMES)))
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=list(CLASS_NAMES),
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    return accuracy, precision, recall, macro_f1, report, matrix


def save_classification_report(report_path: Path, report: str) -> None:
    """Write the sklearn classification report to a text file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")


def save_confusion_matrix(matrix_path: Path, matrix: Any) -> None:
    """Render a labeled confusion matrix as a PNG using Pillow."""
    cell_size = 100
    margin_left = 150
    margin_top = 100
    image_size = (
        margin_left + cell_size * len(CLASS_NAMES) + 20,
        margin_top + cell_size * len(CLASS_NAMES) + 20,
    )
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    maximum = max(int(matrix.max()), 1)

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = int(matrix[row, column])
            intensity = int(255 - (value / maximum) * 200)
            color = (intensity, intensity, 255)
            left = margin_left + column * cell_size
            top = margin_top + row * cell_size
            draw.rectangle(
                (left, top, left + cell_size, top + cell_size),
                fill=color,
                outline="black",
            )
            text = str(value)
            draw.text((left + cell_size // 2, top + cell_size // 2), text, fill="black", anchor="mm")

    for index, class_name in enumerate(CLASS_NAMES):
        center = margin_left + index * cell_size + cell_size // 2
        draw.text((center, margin_top - 25), class_name, fill="black", anchor="mm")
        center = margin_top + index * cell_size + cell_size // 2
        draw.text((margin_left - 15, center), class_name, fill="black", anchor="rm")

    draw.text((margin_left + (cell_size * len(CLASS_NAMES)) // 2, 20), "Predicted", fill="black", anchor="mm")
    draw.text((20, margin_top + (cell_size * len(CLASS_NAMES)) // 2), "Actual", fill="black", anchor="mm")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(matrix_path)


def evaluate(
    checkpoint_path: Path,
    test_dir: Path,
    report_path: Path,
    matrix_path: Path,
) -> None:
    """Evaluate the checkpoint exclusively against the held-out test split."""
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    dataset = load_test_dataset(test_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = load_model(checkpoint, device)
    targets, predictions = run_inference(model, loader, device)
    accuracy, precision, recall, macro_f1, report, matrix = calculate_metrics(
        targets,
        predictions,
    )

    save_classification_report(report_path, report)
    save_confusion_matrix(matrix_path, matrix)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification report:")
    print(report)
    print(f"Classification report disimpan ke {report_path}")
    print(f"Confusion matrix disimpan ke {matrix_path}")


def main() -> int:
    """Run test-set evaluation using paths relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        evaluate(
            checkpoint_path=project_root / "models" / "best_model.pth",
            test_dir=project_root / "dataset" / "processed" / "test",
            report_path=project_root / "reports" / "classification_report.txt",
            matrix_path=project_root / "reports" / "confusion_matrix.png",
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"Evaluasi gagal: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())