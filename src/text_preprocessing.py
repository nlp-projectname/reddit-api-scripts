import re

import nltk
import pandas as pd
from emot.emo_unicode import UNICODE_EMOJI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils import contractions_dict

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))


def expand_contractions(text: str) -> str:
    contractions_re = re.compile("(%s)" % "|".join(contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


# TODO: check this is the desired behavior, otherwise simply eliminate
def convert_emojis(text: str) -> str:
    for emot in UNICODE_EMOJI:
        text = text.replace(
            emot,
            " ".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()),
        )
    return text.replace("_", " ")


def preprocess_text(text: str) -> str:
    # print("Original:", text)
    text = text.lower()
    # print("Lowercased:", text)
    text = expand_contractions(text)
    # print("Contractions expanded:", text)
    text = convert_emojis(text)
    # print("Emojis converted:", text)
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # remove URLs or usernames
    # print("URLs removed:", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove chars, punctuation and numbers
    # print("Chars removed:", text)
    tokens = word_tokenize(text)
    # print("Tokenized:", tokens)
    tokens = [word for word in tokens if word not in stop_words]
    # print("Stopwords removed:", tokens)
    tokens = [PorterStemmer().stem(word) for word in tokens]
    # print("Stemmed:", tokens)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    # print("Lemmatized:", tokens)
    text = " ".join(tokens)
    # print("Final text:", text)
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["content"] = df["content"].apply(preprocess_text)
    return df
