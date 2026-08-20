# clean_text, predict_sentiment, search

# ╔════════════════════════════════════════════════════════════╗
# ║ 🧹CLEANING
# ╚════════════════════════════════════════════════════════════╝
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


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

from tf_keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle

with open("../models/tokenizer_full.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200


def predict_sentiment(text: str, model) -> dict:

    new_text_clean = [clean_text(text)]
    new_text_seq = tokenizer.texts_to_sequences(new_text_clean)
    new_text_pad = pad_sequences(
        new_text_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    new_prob = model.predict(new_text_pad, verbose=0)[0]
    new_pred = int(np.argmax(new_prob, axis=1))
    confiance = float(new_prob[new_pred])

    return dict(
        {
            "prediction": new_pred,
            "probability": confiance,
            "confiance": confiance,
        }
    )
