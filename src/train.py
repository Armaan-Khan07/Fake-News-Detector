"""
train.py
Train a TF-IDF + Logistic Regression (and SVM) fake news classifier.

Usage:
    python src/train.py --true data/True.csv --fake data/Fake.csv

Or, if you already have a single labeled CSV with columns [text, label]:
    python src/train.py --data data/news.csv
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import clean_text, combine_title_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_kaggle_style(true_path, fake_path):
    """Kaggle 'Fake and Real News' dataset: two separate CSVs (True.csv / Fake.csv)."""
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df["label"] = 1  # 1 = REAL
    fake_df["label"] = 0  # 0 = FAKE
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = combine_title_text(df)
    return df[["content", "label"]].rename(columns={"content": "text"})


def load_single_csv(data_path):
    df = pd.read_csv(data_path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"
    return df[["text", "label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", type=str, help="Path to True.csv (Kaggle format)")
    parser.add_argument("--fake", type=str, help="Path to Fake.csv (Kaggle format)")
    parser.add_argument("--data", type=str, help="Path to single CSV with text,label columns")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm"])
    args = parser.parse_args()

    if args.data:
        df = load_single_csv(args.data)
    elif args.true and args.fake:
        df = load_kaggle_style(args.true, args.fake)
    else:
        raise ValueError("Provide either --data OR both --true and --fake")

    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    print("Cleaning text (this can take a minute on large datasets)...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    if args.model == "logreg":
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    else:
        clf = LinearSVC(class_weight="balanced")

    print(f"Training {args.model}...")
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({args.model})")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    print(f"Saved confusion matrix to {MODEL_DIR}/confusion_matrix.png")

    joblib.dump(clf, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print(f"Saved model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
