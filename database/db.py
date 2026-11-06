"""SQLite persistence for prediction history."""

from __future__ import annotations

import sqlite3
import csv
import io
from pathlib import Path
from typing import Any


def initialize_database(database_path: Path) -> None:
    """Create the prediction history table when it does not exist."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
                actual_class TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
            """
        )


def save_prediction(
    database_path: Path,
    filename: str,
    predicted_class: str,
    confidence: float,
) -> int:
    """Store one prediction in the local history database."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (filename, predicted_class, confidence)
            VALUES (?, ?, ?)
            """,
            (filename, predicted_class, confidence),
        )
        return int(cursor.lastrowid)


def get_recent_predictions(
    database_path: Path,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent prediction history entries."""
    if limit <= 0:
        raise ValueError("limit harus lebih besar dari nol.")
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT created_at, filename, predicted_class, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_feedback(
    database_path: Path,
    prediction_id: int,
    is_correct: bool,
    actual_class: str | None = None,
) -> None:
    """Store user feedback for a saved prediction."""
    if prediction_id <= 0:
        raise ValueError("prediction_id harus lebih besar dari nol.")
    if not is_correct and not actual_class:
        raise ValueError("Kategori aktual wajib diisi untuk prediksi yang salah.")

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_feedback
                (prediction_id, is_correct, actual_class)
            VALUES (?, ?, ?)
            """,
            (prediction_id, int(is_correct), actual_class),
        )


def get_prediction_summary(database_path: Path) -> dict[str, Any]:
    """Return aggregate prediction and feedback statistics."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        prediction_stats = connection.execute(
            """
            SELECT COUNT(*), COALESCE(AVG(confidence), 0.0)
            FROM predictions
            """
        ).fetchone()
        feedback_stats = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(is_correct), 0)
            FROM prediction_feedback
            """
        ).fetchone()
        category_rows = connection.execute(
            """
            SELECT predicted_class, COUNT(*) AS count
            FROM predictions
            GROUP BY predicted_class
            ORDER BY count DESC
            """
        ).fetchall()

    prediction_count, average_confidence = prediction_stats
    feedback_count, correct_feedback_count = feedback_stats
    feedback_accuracy = (
        correct_feedback_count / feedback_count if feedback_count else None
    )
    return {
        "prediction_count": int(prediction_count),
        "average_confidence": float(average_confidence),
        "feedback_count": int(feedback_count),
        "feedback_accuracy": feedback_accuracy,
        "category_counts": {
            str(category): int(count) for category, count in category_rows
        },
    }


def export_feedback_csv(database_path: Path) -> bytes:
    """Return prediction feedback metadata as UTF-8 CSV bytes."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                p.created_at,
                p.filename,
                p.predicted_class,
                p.confidence,
                f.is_correct,
                f.actual_class,
                f.created_at AS feedback_created_at
            FROM prediction_feedback AS f
            JOIN predictions AS p ON p.id = f.prediction_id
            ORDER BY f.id ASC
            """
        ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "prediction_created_at",
            "filename",
            "predicted_class",
            "confidence",
            "is_correct",
            "actual_class",
            "feedback_created_at",
        ]
    )
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")