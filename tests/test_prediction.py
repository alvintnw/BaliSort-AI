from __future__ import annotations

import io

import pytest
import torch
from PIL import Image
from torch import nn

from database.db import (
    export_feedback_csv,
    get_prediction_summary,
    get_recent_predictions,
    save_feedback,
    save_prediction,
)
from src.model import CLASS_NAMES
from src.predict import open_uploaded_image, predict_image
from src.recommendation import get_recommendation


class FixedModel(nn.Module):
    """Small deterministic model used to test inference behavior."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return fixed logits with one output per waste class."""
        return self.logits.expand(inputs.size(0), -1)


def image_bytes(format_name: str = "PNG") -> bytes:
    """Create a small in-memory image for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(20, 120, 40)).save(buffer, format=format_name)
    return buffer.getvalue()


def test_open_uploaded_image_converts_to_rgb() -> None:
    image = open_uploaded_image(image_bytes())

    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_open_uploaded_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="bukan gambar"):
        open_uploaded_image(b"not an image")


def test_predict_image_returns_class_confidence_and_probabilities() -> None:
    model = FixedModel([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    image = Image.open(io.BytesIO(image_bytes()))

    result = predict_image(model, image, torch.device("cpu"))

    assert result["class_name"] == "cardboard"
    assert result["confidence"] > 0.6
    assert result["is_uncertain"] is False
    assert set(result["probabilities"]) == set(CLASS_NAMES)


def test_predict_image_marks_low_confidence_prediction_uncertain() -> None:
    model = FixedModel([0.0] * len(CLASS_NAMES))
    image = Image.open(io.BytesIO(image_bytes()))

    result = predict_image(model, image, torch.device("cpu"))

    assert result["is_uncertain"] is True


def test_recommendation_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Kategori tidak dikenal"):
        get_recommendation("unknown")


def test_prediction_history_round_trip(tmp_path) -> None:
    database_path = tmp_path / "predictions.db"
    prediction_id = save_prediction(database_path, "sample.jpg", "plastic", 0.875)

    history = get_recent_predictions(database_path)

    assert prediction_id > 0
    assert len(history) == 1
    assert history[0]["filename"] == "sample.jpg"
    assert history[0]["predicted_class"] == "plastic"
    assert history[0]["confidence"] == pytest.approx(0.875)


def test_feedback_requires_actual_class_when_prediction_is_wrong(tmp_path) -> None:
    database_path = tmp_path / "predictions.db"
    prediction_id = save_prediction(database_path, "sample.jpg", "plastic", 0.875)

    with pytest.raises(ValueError, match="Kategori aktual"):
        save_feedback(database_path, prediction_id, False)


def test_feedback_can_be_saved_for_prediction(tmp_path) -> None:
    database_path = tmp_path / "predictions.db"
    prediction_id = save_prediction(database_path, "sample.jpg", "plastic", 0.875)

    save_feedback(database_path, prediction_id, False, "glass")

    import sqlite3

    with sqlite3.connect(database_path) as connection:
        feedback = connection.execute(
            "SELECT is_correct, actual_class FROM prediction_feedback"
        ).fetchone()
    assert feedback == (0, "glass")


def test_prediction_summary_aggregates_predictions_and_feedback(tmp_path) -> None:
    database_path = tmp_path / "predictions.db"
    first_id = save_prediction(database_path, "one.jpg", "plastic", 0.8)
    second_id = save_prediction(database_path, "two.jpg", "glass", 0.6)
    save_feedback(database_path, first_id, True)
    save_feedback(database_path, second_id, False, "paper")

    summary = get_prediction_summary(database_path)

    assert summary["prediction_count"] == 2
    assert summary["average_confidence"] == pytest.approx(0.7)
    assert summary["feedback_count"] == 2
    assert summary["feedback_accuracy"] == pytest.approx(0.5)
    assert summary["category_counts"] == {"glass": 1, "plastic": 1}


def test_feedback_csv_exports_metadata_without_image_data(tmp_path) -> None:
    database_path = tmp_path / "predictions.db"
    prediction_id = save_prediction(database_path, "sample.jpg", "plastic", 0.8)
    save_feedback(database_path, prediction_id, False, "glass")

    exported = export_feedback_csv(database_path).decode("utf-8-sig")

    assert "filename,predicted_class,confidence" in exported
    assert "sample.jpg,plastic,0.8,0,glass" in exported
    assert "image_bytes" not in exported


def test_prediction_history_rejects_non_positive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        get_recent_predictions(tmp_path / "predictions.db", limit=0)