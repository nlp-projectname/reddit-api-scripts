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


class TextPreprocessor:
    def __init__(
        self,
        lowercase=True,
        expand_contractions=True,
        convert_emojis=True,
        remove_urls=True,
        remove_punctuation=True,
        remove_stopwords=True,
        apply_stemming=True,
        apply_lemmatization=True,
    ):
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions
        self.convert_emojis = convert_emojis
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def _expand_contractions(self, text: str) -> str:
        contractions_re = re.compile(
            r"\b(" + "|".join(contractions_dict.keys()) + r")\b"
        )

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    def _convert_emojis(self, text: str) -> str:
        for emot in UNICODE_EMOJI:
            text = text.replace(
                emot,
                " ".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()),
            )
        return text.replace("_", " ")

    def preprocess_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            text = self._expand_contractions(text)
        if self.convert_emojis:
            text = self._convert_emojis(text)
        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # remove URLs or usernames
        if self.remove_punctuation:
            text = re.sub(
                r"[^A-Za-z\s]", "", text
            )  # remove chars, punctuation and numbers
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        if self.apply_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        if self.apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_column: str = "content"
    ) -> pd.DataFrame:
        df[text_column] = df[text_column].apply(self.preprocess_text)
        return df
