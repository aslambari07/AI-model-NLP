from __future__ import annotations

from pathlib import Path

import joblib


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan `python src/train.py` terlebih dahulu."
        )
    return joblib.load(MODEL_PATH)


def predict_text(text: str) -> dict:
    model = load_model()
    probabilities = model.predict_proba([text])[0]
    best_index = int(probabilities.argmax())

    return {
        "label": LABELS[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            LABELS[index]: float(score) for index, score in enumerate(probabilities)
        },
    }
