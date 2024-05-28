from enum import Enum
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sp
import word2vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractionMethod(Enum):
    BOW = "bow"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"


class FeatureExtractor:
    def __init__(
        self, method: FeatureExtractionMethod, max_features: Optional[int] = None
    ):
        self.method = method
        self.max_features = max_features
        self.vectorizer: Optional[Union[CountVectorizer, TfidfVectorizer]] = None
        self.model: Optional[word2vec.WordVectors] = None

    def fit_transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr.csr_matrix]:
        if self.method == FeatureExtractionMethod.BOW:
            self.vectorizer = CountVectorizer(max_features=self.max_features)
            return self.vectorizer.fit_transform(texts)
        elif self.method == FeatureExtractionMethod.TFIDF:
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
            return self.vectorizer.fit_transform(texts)
        elif self.method == FeatureExtractionMethod.WORD2VEC:
            with open("texts.txt", "w") as f:
                for text in texts:
                    f.write("%s\n" % text)
            word2vec.word2vec(
                "texts.txt",
                "word2vec.bin",
                size=self.max_features if self.max_features else 100,
                verbose=True,
            )
            self.model = word2vec.load("word2vec.bin")
            sentences = [text.split() for text in texts]
            if self.model is None:
                raise ValueError("Word2Vec model not initialized")
            return np.array(
                [
                    np.mean(
                        [self.model[word] for word in words if word in self.model]
                        or [np.zeros(self.max_features)],
                        axis=0,
                    )
                    for words in sentences
                ]
            )
        else:
            raise ValueError("Unsupported feature extraction method")

    def transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr.csr_matrix]:
        if self.method in [FeatureExtractionMethod.BOW, FeatureExtractionMethod.TFIDF]:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted")
            return self.vectorizer.transform(texts)
        elif self.method == FeatureExtractionMethod.WORD2VEC:
            sentences = [text.split() for text in texts]
            if self.model is None:
                raise ValueError("Word2Vec model not initialized")
            return np.array(
                [
                    np.mean(
                        [self.model[word] for word in words if word in self.model]
                        or [np.zeros(self.max_features)],
                        axis=0,
                    )
                    for words in sentences
                ]
            )
        else:
            raise ValueError("Unsupported feature extraction method")


# Example usage
if __name__ == "__main__":
    texts = [
        "I love machine learning",
        "Natural language processing is fascinating",
        "Word2Vec is a great tool for word embeddings",
    ]

    extractor = FeatureExtractor(FeatureExtractionMethod.WORD2VEC, max_features=100)
    features = extractor.fit_transform(texts)
    print(features)
