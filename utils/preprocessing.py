import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)

STOP_FR = set(stopwords.words("french"))
STOP_EN = set(stopwords.words("english"))
STOP_ALL = STOP_FR.union(STOP_EN)
STOP_ALL.add("generique")

STEMMER = SnowballStemmer("french")


def clean_text(text) -> str:
    if pd.isnull(text):
        return ""

    text = re.sub(r"<.*?>", " ", str(text))
    text = text.replace("<br />", " ")

    text = text.replace("&amp;", "")
    text = text.replace("&nbsp;", "")
    text = text.replace("&lt;", "")
    text = text.replace("&gt;", "")
    text = text.replace("&quot;", "")
    text = text.replace("&#39;", "")
    text = text.replace("&eacute;", "e")
    text = text.replace("&egrave;", "e")
    text = text.replace("&ecirc;", "e")

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    tokens = [w for w in text.split() if w not in STOP_ALL and len(w) > 1]
    return " ".join(tokens)


def stem_text(text: str) -> str:
    tokens = [STEMMER.stem(w) for w in text.split()]
    return " ".join(tokens)


def preprocess_product_text(designation: str, description: str) -> str:
    clean_designation = clean_text(designation)
    clean_description = clean_text(description)

    full_text = f"{clean_designation} {clean_description}".strip()
    full_text = remove_stopwords(full_text)
    full_text = stem_text(full_text)

    return full_text