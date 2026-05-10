from __future__ import annotations

import json
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def clean_text(text: str) -> str:
    return " ".join(text.split())


def load_data():
    dataset = load_dataset("ag_news")

    train_texts = [clean_text(text) for text in dataset["train"]["text"]]
    test_texts = [clean_text(text) for text in dataset["test"]["text"]]

    y_train = dataset["train"]["label"]
    y_test = dataset["test"]["label"]
    return train_texts, test_texts, y_train, y_test


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=120000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def main():
    print("Mengunduh dan menyiapkan dataset...")
    x_train, x_test, y_train, y_test = load_data()

    print("Melatih model...")
    model = build_pipeline()
    model.fit(x_train, y_train)

    print("Melakukan evaluasi...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=[LABELS[i] for i in range(len(LABELS))],
        output_dict=True,
    )

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Akurasi test: {accuracy:.4f}")
    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Metrik tersimpan di: {METRICS_PATH}")


if __name__ == "__main__":
    main()
