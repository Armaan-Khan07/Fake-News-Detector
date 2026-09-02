"""
predict.py
Load the trained model + vectorizer and predict on new text.

Usage:
    python src/predict.py "Some news article text here..."
"""

import sys
import os
import joblib
from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    # Confidence score if the model supports it
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vec).max())
    elif hasattr(model, "decision_function"):
        import numpy as np
        score = model.decision_function(vec)[0]
        confidence = float(1 / (1 + pow(2.718281828, -abs(score))))  # sigmoid approx

    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "news text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    label, confidence = predict(text)
    conf_str = f" (confidence: {confidence:.2%})" if confidence else ""
    print(f"Prediction: {label}{conf_str}")
