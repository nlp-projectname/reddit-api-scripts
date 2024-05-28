from enum import Enum
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sp
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.cli import download


class FeatureExtractionMethod(Enum):
    BOW = "bow"
    TFIDF = "tfidf"
    GLOVE = "glove"


class FeatureExtractor:
    def __init__(
        self, method: FeatureExtractionMethod, max_features: Optional[int] = None
    ):
        self.method = method
        self.max_features = max_features
        self.vectorizer: Optional[Union[CountVectorizer, TfidfVectorizer]] = None
        if self.method == FeatureExtractionMethod.GLOVE:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except OSError:
                print("Spacy model 'en_core_web_md' not found. Installing...")
                download("en_core_web_md")
                self.nlp = spacy.load("en_core_web_md")

    def fit_transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        if self.method == FeatureExtractionMethod.BOW:
            self.vectorizer = CountVectorizer(max_features=self.max_features)
            return self.vectorizer.fit_transform(texts)
        elif self.method == FeatureExtractionMethod.TFIDF:
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
            return self.vectorizer.fit_transform(texts)
        elif self.method == FeatureExtractionMethod.GLOVE:
            return np.array([self.nlp(text).vector for text in texts])
        else:
            raise ValueError("Unsupported feature extraction method")

    def transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        if self.method in [FeatureExtractionMethod.BOW, FeatureExtractionMethod.TFIDF]:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted")
            return self.vectorizer.transform(texts)
        elif self.method == FeatureExtractionMethod.GLOVE:
            return np.array([self.nlp(text).vector for text in texts])
        else:
            raise ValueError("Unsupported feature extraction method")


# Example usage
if __name__ == "__main__":
    texts = [
        "I love machine learning",
    ]

    extractor = FeatureExtractor(FeatureExtractionMethod.GLOVE)
    features = extractor.fit_transform(texts)
    print(features)
