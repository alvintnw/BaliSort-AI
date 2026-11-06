"""Training pipeline for BaliSort AI MobileNetV3-Large classification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    from src.model import CLASS_NAMES, build_model
    from src.preprocessing import get_eval_transforms, get_train_transforms
except ImportError:
    from model import CLASS_NAMES, build_model
    from preprocessing import get_eval_transforms, get_train_transforms


DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0001
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics produced by one training or validation epoch."""

    loss: float
    accuracy: float
    macro_f1: float


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible dataset ordering and optimization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA when available, otherwise warn and use CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Menggunakan GPU: {torch.cuda.get_device_name(device)}")
        return device

    print("CUDA tidak tersedia; training dilanjutkan menggunakan CPU.")
    return torch.device("cpu")


def validate_class_mapping(
    class_to_idx: dict[str, int],
    dataset_name: str,
) -> None:
    """Ensure a dataset has exactly the required class names and indices."""
    expected_mapping = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected_mapping:
        raise ValueError(
            f"Mapping kelas {dataset_name} tidak sesuai. "
            f"Diharapkan {expected_mapping}, ditemukan {class_to_idx}."
        )


def load_datasets(
    dataset_root: Path,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Load only the training and validation ImageFolder datasets."""
    train_dir = dataset_root / "train"
    validation_dir = dataset_root / "validation"
    for directory in (train_dir, validation_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {directory}")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    validation_dataset = datasets.ImageFolder(
        validation_dir,
        transform=get_eval_transforms(),
    )
    validate_class_mapping(train_dataset.class_to_idx, "training")
    validate_class_mapping(validation_dataset.class_to_idx, "validation")
    return train_dataset, validation_dataset


def calculate_class_weights(
    targets: list[int],
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Calculate inverse-frequency weights for CrossEntropyLoss."""
    counts = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes)
    if torch.any(counts == 0):
        missing = [str(index) for index, count in enumerate(counts.tolist()) if count == 0]
        raise ValueError(f"Kelas tanpa gambar training: {', '.join(missing)}")

    weights = counts.sum().float() / (num_classes * counts.float())
    return weights.to(device)


def calculate_macro_f1(confusion_matrix: Tensor) -> float:
    """Calculate unweighted macro F1 from a multiclass confusion matrix."""
    true_positive = confusion_matrix.diag().float()
    precision_denominator = confusion_matrix.sum(dim=0).float()
    recall_denominator = confusion_matrix.sum(dim=1).float()
    precision = true_positive / precision_denominator.clamp_min(1.0)
    recall = true_positive / recall_denominator.clamp_min(1.0)
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-12)
    return float(f1.mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler,
    num_classes: int,
) -> EpochMetrics:
    """Run one training or evaluation epoch and return loss, accuracy, and F1."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = outputs.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            indices = targets * num_classes + predictions
            confusion_matrix += torch.bincount(
                indices,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

    if total_samples == 0:
        raise ValueError("Dataset kosong; tidak dapat menjalankan epoch.")

    accuracy = float(confusion_matrix.diag().sum().item() / total_samples)
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=accuracy,
        macro_f1=calculate_macro_f1(confusion_matrix),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    metrics: EpochMetrics,
    configuration: dict[str, Any],
) -> None:
    """Save model state and reproducibility metadata for the best epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": list(CLASS_NAMES),
            "configuration": configuration,
            "metrics": {
                "validation_loss": metrics.loss,
                "validation_accuracy": metrics.accuracy,
                "validation_macro_f1": metrics.macro_f1,
            },
        },
        path,
    )


def train_model(
    dataset_root: Path,
    checkpoint_path: Path,
    pretrained: bool = True,
) -> None:
    """Train MobileNetV3 using train/validation data and early stopping."""
    set_random_seed()
    device = get_device()
    train_dataset, validation_dataset = load_datasets(dataset_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    class_weights = calculate_class_weights(
        train_dataset.targets,
        len(CLASS_NAMES),
        device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    configuration: dict[str, Any] = {
        "architecture": "MobileNetV3-Large",
        "pretrained": pretrained,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "class_weights": class_weights.detach().cpu().tolist(),
    }

    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, len(CLASS_NAMES)
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, None, scaler, len(CLASS_NAMES)
        )
        scheduler.step(validation_metrics.loss)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss: {train_metrics.loss:.4f}, "
            f"train accuracy: {train_metrics.accuracy:.4f}, "
            f"train macro F1: {train_metrics.macro_f1:.4f}, "
            f"val loss: {validation_metrics.loss:.4f}, "
            f"val accuracy: {validation_metrics.accuracy:.4f}, "
            f"val macro F1: {validation_metrics.macro_f1:.4f}"
        )

        if validation_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = validation_metrics.macro_f1
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
                configuration,
            )
            print(f"  Checkpoint terbaik disimpan ke {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation macro F1 tidak membaik.")
                break


def main() -> int:
    """Run training from paths relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        train_model(
            dataset_root=project_root / "dataset" / "processed",
            checkpoint_path=project_root / "models" / "best_model.pth",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Training gagal: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())