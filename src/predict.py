"""Inference helpers for the BaliSort AI Streamlit application."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from src.model import CLASS_NAMES, build_model
from src.preprocessing import get_eval_transforms


CONFIDENCE_THRESHOLD = 0.60


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the trained MobileNetV3 checkpoint and validate its class order."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Format checkpoint tidak valid.")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Urutan kelas checkpoint tidak sesuai.")

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def validate_image(image: Image.Image) -> Image.Image:
    """Validate and normalize an uploaded image as RGB."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Gambar memiliki ukuran tidak valid.")
    return image.convert("RGB")


def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
) -> dict[str, Any]:
    """Predict one image and return label, confidence, and all probabilities."""
    try:
        image = validate_image(image)
    except (AttributeError, ValueError) as error:
        raise ValueError("File yang diunggah bukan gambar valid.") from error

    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    confidence, index = probabilities.max(dim=0)
    confidence_value = float(confidence.item())
    return {
        "class_name": CLASS_NAMES[index.item()],
        "confidence": confidence_value,
        "is_uncertain": confidence_value < CONFIDENCE_THRESHOLD,
        "probabilities": {
            class_name: float(probabilities[class_index].item())
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
    }


def open_uploaded_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes and raise a user-facing error for invalid files."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return validate_image(image)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError("File bukan gambar yang valid atau rusak.") from error