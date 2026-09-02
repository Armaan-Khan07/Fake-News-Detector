"""
preprocess.py
Text cleaning utilities for the fake news detection project.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only runs once, then cached)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/HTML/punctuation/numbers, remove stopwords,
    and lemmatize. Returns a cleaned string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                           # HTML tags
    text = re.sub(r"\S+@\S+", " ", text)                         # emails
    text = re.sub(r"[^a-z\s]", " ", text)                        # numbers/punct
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def combine_title_text(df, title_col="title", text_col="text"):
    """Merge title + article body into a single 'content' column (title carries a lot of signal)."""
    df = df.copy()
    df["content"] = (
        df.get(title_col, "").fillna("") + " " + df.get(text_col, "").fillna("")
    )
    return df


if __name__ == "__main__":
    sample = "BREAKING!!! Scientists find <b>SHOCKING</b> truth at https://fake.com — read NOW!!!"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
