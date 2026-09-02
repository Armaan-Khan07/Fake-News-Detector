"""
streamlit_app.py
Simple web UI for the fake news detector.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import streamlit as st

# allow importing src/preprocess.py regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vec).max())
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        confidence = float(1 / (1 + pow(2.718281828, -abs(score))))

    return ("REAL" if pred == 1 else "FAKE"), confidence


st.title("📰 Fake News Detector")
st.caption("TF-IDF + Logistic Regression classifier")

model, vectorizer = load_artifacts()

if model is None:
    st.error(
        "No trained model found in `models/`. Train one first:\n\n"
        "`python src/train.py --true data/True.csv --fake data/Fake.csv`"
    )
else:
    text_input = st.text_area(
        "Paste a news headline or article:",
        height=200,
        placeholder="e.g. Government announces new policy on renewable energy...",
    )

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            label, confidence = predict(text_input, model, vectorizer)
            conf_pct = f"{confidence:.1%}" if confidence else "N/A"

            if label == "REAL":
                st.success(f"✅ Predicted: **REAL** (confidence: {conf_pct})")
            else:
                st.error(f"🚨 Predicted: **FAKE** (confidence: {conf_pct})")

            with st.expander("See cleaned text used for prediction"):
                st.code(clean_text(text_input))

st.divider()
st.caption(
    "⚠️ This is a portfolio ML project, not a fact-checking authority. "
    "Predictions reflect linguistic patterns learned from training data, not verified truth."
)
