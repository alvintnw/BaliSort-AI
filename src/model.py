"""MobileNetV3 model construction for BaliSort AI image classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


CLASS_NAMES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)


def _validate_model_parameters(
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> None:
    """Validate arguments used to construct the classifier."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes harus berupa integer.")
    if num_classes <= 0:
        raise ValueError("num_classes harus lebih besar dari nol.")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained harus berupa boolean.")
    if not isinstance(freeze_backbone, bool):
        raise TypeError("freeze_backbone harus berupa boolean.")


def build_model(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a MobileNetV3-Large classifier for the requested class count.

    The returned model remains device-agnostic and can be moved to CPU or CUDA
    with ``model.to(device)``. When ``freeze_backbone`` is enabled, only the
    replacement classifier parameters have ``requires_grad=True``.
    """
    _validate_model_parameters(num_classes, pretrained, freeze_backbone)

    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    try:
        model = mobilenet_v3_large(weights=weights)
    except (OSError, RuntimeError) as error:
        raise RuntimeError("Gagal memuat arsitektur atau bobot MobileNetV3.") from error

    if not isinstance(model.classifier, nn.Sequential) or not model.classifier:
        raise RuntimeError("Struktur classifier MobileNetV3 tidak sesuai.")

    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise RuntimeError("Lapisan classifier terakhir bukan nn.Linear.")

    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model
