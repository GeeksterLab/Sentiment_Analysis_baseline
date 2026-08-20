# clean_text, predict_sentiment, search

# ╔════════════════════════════════════════════════════════════╗
# ║ 🧹CLEANING
# ╚════════════════════════════════════════════════════════════╝
import re
import pickle
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from core.config import settings

NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}

for package, resource_path in NLTK_PACKAGES.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)  # delete HTML balise
    text = re.sub(r"http\S+", "", text)  # delete URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # delete ponctuation
    text = re.sub(r"\s+", " ", text)  # delete duplicated spaces

    tokens = word_tokenize(text)
    stop_words_en = set(stopwords.words("english"))
    stop_words_fr = set(stopwords.words("french"))
    filtered_token = [
        word
        for word in tokens
        if word not in stop_words_en and word not in stop_words_fr
    ]
    text = " ".join(filtered_token)

    return text


# ╔════════════════════════════════════════════════════════════╗
# ║ 🔮 PREDICTION
# ╚════════════════════════════════════════════════════════════╝
import numpy as np

MAX_LEN = 200


@lru_cache(maxsize=1)
def get_model():
    from tf_keras.models import load_model

    return load_model(settings.MODEL_PATH)


@lru_cache(maxsize=1)
def get_tokenizer():
    with open(settings.TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


def predict_sentiment(text: str) -> dict:
    from tf_keras.preprocessing.sequence import pad_sequences

    model = get_model()
    tokenizer = get_tokenizer()
    new_text_clean = [clean_text(text)]
    new_text_seq = tokenizer.texts_to_sequences(new_text_clean)
    new_text_pad = pad_sequences(
        new_text_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    new_prob = model.predict(new_text_pad, verbose=0)[0]
    new_pred = int(np.argmax(new_prob))
    confiance = float(new_prob[new_pred])

    return dict(
        {
            "prediction": new_pred,
            "probability": confiance,
            "confiance": confiance,
        }
    )
