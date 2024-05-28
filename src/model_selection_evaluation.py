from enum import Enum
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sp
import spacy
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from spacy.cli import download
from tensorflow.keras.layers import LSTM, Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class ModelType(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    RNN = "rnn"
    LSTM = "lstm"


class ModelTrainer:
    def __init__(
        self, model_type: ModelType, random_state: int = 42, max_seq_length: int = 100
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.max_seq_length = max_seq_length
        self.model = self._initialize_model()
        self.tokenizer = None

    def _initialize_model(self):
        if self.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(random_state=self.random_state)
        elif self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == ModelType.SVM:
            return SVC(random_state=self.random_state)
        elif self.model_type == ModelType.RNN:
            model = Sequential()
            model.add(Embedding(input_dim=5000, output_dim=128))
            model.add(SimpleRNN(128))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            return model
        elif self.model_type == ModelType.LSTM:
            model = Sequential()
            model.add(Embedding(input_dim=5000, output_dim=128))
            model.add(LSTM(128))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            return model
        else:
            raise ValueError("Unsupported model type")

    def fit_transform(self, texts: List[str], labels: np.ndarray):
        if self.model_type in [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.SVM,
        ]:
            self.vectorizer = CountVectorizer(max_features=self.max_seq_length)
            features = self.vectorizer.fit_transform(texts)
            self.model.fit(features, labels)
            return features
        elif self.model_type in [ModelType.RNN, ModelType.LSTM]:
            self.tokenizer = Tokenizer(num_words=5000)
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initiated correctly")
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_seq_length)
            self.model.fit(
                padded_sequences, labels, epochs=5, batch_size=32, validation_split=0.2
            )
            return padded_sequences
        else:
            raise ValueError("Unsupported feature extraction method")

    def transform(self, texts: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        if self.model_type in [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.SVM,
        ]:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted")
            return self.vectorizer.transform(texts)
        elif self.model_type in [ModelType.RNN, ModelType.LSTM]:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not fitted")
            sequences = self.tokenizer.texts_to_sequences(texts)
            return pad_sequences(sequences, maxlen=self.max_seq_length)
        else:
            raise ValueError("Unsupported feature extraction method")

    def evaluate(self, texts: List[str], labels: np.ndarray) -> dict:
        if self.model_type in [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.SVM,
        ]:
            features = self.transform(texts)
            y_pred = self.model.predict(features)
            accuracy = accuracy_score(labels, y_pred)
            report = classification_report(labels, y_pred, output_dict=True)
            return {"accuracy": accuracy, "report": report}
        elif self.model_type in [ModelType.RNN, ModelType.LSTM]:
            sequences = self.transform(texts)
            loss, accuracy = self.model.evaluate(sequences, labels)
            return {"accuracy": accuracy, "loss": loss}
        else:
            raise ValueError("Unsupported feature extraction method")

    def predict(self, texts: List[str]) -> np.ndarray:
        features = self.transform(texts)
        return self.model.predict(features)
