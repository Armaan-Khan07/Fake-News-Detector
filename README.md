# 📰 Fake News Detection

A machine learning project that classifies news articles as **REAL** or **FAKE** using TF-IDF vectorization and classic ML models (Logistic Regression / Linear SVM). Includes a Streamlit web app for interactive predictions.

## Project Structure

```
fake-news-detection/
├── data/
│   ├── sample_True.csv       # small sample dataset (for testing the pipeline)
│   ├── sample_Fake.csv       # small sample dataset (for testing the pipeline)
│   ├── True.csv               # <- put full Kaggle dataset here
│   └── Fake.csv               # <- put full Kaggle dataset here
├── src/
│   ├── preprocess.py          # text cleaning (stopwords, lemmatization, etc.)
│   ├── train.py                # trains TF-IDF + LogReg/SVM, saves model
│   └── predict.py             # CLI prediction on new text
├── app/
│   └── streamlit_app.py       # web UI
├── models/                    # saved model.pkl, vectorizer.pkl, confusion_matrix.png
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

This project is built for the **Kaggle Fake and Real News Dataset**:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

1. Download `True.csv` and `Fake.csv` from the link above (free Kaggle account needed).
2. Place both files inside the `data/` folder.

**Don't have Kaggle access yet?** The repo ships with `sample_True.csv` and `sample_Fake.csv` — 10 rows each — so you can test the full pipeline immediately. Swap in the real files later; nothing else changes.

## 3. Train the model

Using the sample data (quick test):
```bash
python src/train.py --true data/sample_True.csv --fake data/sample_Fake.csv --model logreg
```

Using the full Kaggle dataset:
```bash
python src/train.py --true data/True.csv --fake data/Fake.csv --model logreg
# or try SVM:
python src/train.py --true data/True.csv --fake data/Fake.csv --model svm
```

This will:
- Clean and preprocess all article text
- Vectorize with TF-IDF (unigrams + bigrams)
- Train a Logistic Regression or Linear SVM classifier
- Print accuracy, precision/recall/F1
- Save a confusion matrix plot to `models/confusion_matrix.png`
- Save `model.pkl` and `vectorizer.pkl` to `models/`

Expect **~98–99% accuracy** on the full Kaggle dataset (it's a fairly separable dataset — a known caveat for this task, see "Limitations" below).

## 4. Predict on new text

CLI:
```bash
python src/predict.py "Your news headline or article text here"
```

Or launch the web app:
```bash
streamlit run app/streamlit_app.py
```

## 5. How it works

1. **Preprocessing** (`preprocess.py`): lowercase → strip URLs/HTML/numbers/punctuation → remove stopwords → lemmatize.
2. **Feature extraction**: TF-IDF with unigrams + bigrams, top 50,000 features.
3. **Model**: Logistic Regression (`class_weight="balanced"`) or Linear SVM — both fast to train and easy to explain.
4. **Evaluation**: accuracy, classification report, confusion matrix.

## Limitations (good to mention in interviews/portfolio)

- The Kaggle dataset is known to have some **source/style leakage** (e.g. Reuters-style formatting correlates strongly with "REAL"), which inflates accuracy. A model trained on it may just be learning writing style, not truthfulness.
- This is a **linguistic pattern classifier**, not a fact-checker — it can't verify real-world claims.
- For a stronger portfolio story, consider testing the model on out-of-distribution headlines (e.g. recent real news it's never seen) and discussing where it fails.

## Possible extensions

- Swap TF-IDF + LogReg for a fine-tuned **BERT/DistilBERT** model and compare accuracy.
- Add **explainability** (LIME/SHAP) to show which words drove a prediction.
- Deploy the Streamlit app on Streamlit Community Cloud or Hugging Face Spaces.
