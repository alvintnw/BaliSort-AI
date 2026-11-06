"""Waste sorting recommendations for BaliSort AI classes."""

from __future__ import annotations

from typing import Final


RECOMMENDATIONS: Final[dict[str, dict[str, str]]] = {
    "cardboard": {
        "label": "Kardus",
        "bin": "Sampah anorganik / kertas",
        "advice": "Lipat dan pastikan kardus kering serta bebas sisa makanan sebelum didaur ulang.",
    },
    "glass": {
        "label": "Kaca",
        "bin": "Sampah anorganik / kaca",
        "advice": "Pisahkan kaca dari sampah lain dan bungkus pecahan dengan aman.",
    },
    "metal": {
        "label": "Logam",
        "bin": "Sampah anorganik / logam",
        "advice": "Bersihkan sisa isi lalu pisahkan kaleng atau logam untuk didaur ulang.",
    },
    "paper": {
        "label": "Kertas",
        "bin": "Sampah anorganik / kertas",
        "advice": "Jaga tetap kering dan pisahkan dari kertas berminyak atau terkontaminasi.",
    },
    "plastic": {
        "label": "Plastik",
        "bin": "Sampah anorganik / plastik",
        "advice": "Kosongkan dan bilas kemasan plastik sebelum disetorkan ke bank sampah.",
    },
    "trash": {
        "label": "Residu",
        "bin": "Sampah residu",
        "advice": "Masukkan ke wadah residu; jangan mencampurnya dengan bahan yang masih dapat didaur ulang.",
    },
}


def get_recommendation(class_name: str) -> dict[str, str]:
    """Return a sorting recommendation for a known model class."""
    if class_name not in RECOMMENDATIONS:
        raise ValueError(f"Kategori tidak dikenal: {class_name}")
    return RECOMMENDATIONS[class_name]