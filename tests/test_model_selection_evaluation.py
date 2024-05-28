import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.model_selection_evaluation import ModelTrainer, ModelType
from src.text_preprocessing import TextPreprocessor

texts = [
    "I love machine learning",
    "Natural language processing is fascinating",
    "Deep learning is a subset of machine learning",
    "AI and ML are revolutionizing the tech industry",
]
labels = np.array([1, 0, 1, 0])

preprocessed_texts = [
    "love machin learn",
    "natur languag process fascin",
    "deep learn subset machin learn",
    "ai ml revolut tech industri",
]


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


@pytest.fixture
def model_trainer_logistic():
    return ModelTrainer(ModelType.LOGISTIC_REGRESSION, max_seq_length=100)


@pytest.fixture
def model_trainer_rnn():
    return ModelTrainer(ModelType.RNN, max_seq_length=10)


def test_model_initialization_logistic(model_trainer_logistic):
    assert model_trainer_logistic.model_type == ModelType.LOGISTIC_REGRESSION
    assert model_trainer_logistic.model is not None


def test_model_initialization_rnn(model_trainer_rnn):
    assert model_trainer_rnn.model_type == ModelType.RNN
    assert model_trainer_rnn.model is not None


def test_fit_transform_logistic(model_trainer_logistic):
    features = model_trainer_logistic.fit_transform(preprocessed_texts, labels)
    assert features.shape[0] == len(preprocessed_texts)
    assert features.shape[1] <= model_trainer_logistic.max_seq_length


def test_fit_transform_rnn(model_trainer_rnn):
    features = model_trainer_rnn.fit_transform(preprocessed_texts, labels)
    assert features.shape[0] == len(preprocessed_texts)
    assert features.shape[1] == model_trainer_rnn.max_seq_length


def test_transform_logistic(model_trainer_logistic):
    model_trainer_logistic.fit_transform(preprocessed_texts, labels)
    new_features = model_trainer_logistic.transform(["Deep learning is great"])
    assert new_features.shape[0] == 1
    assert new_features.shape[1] <= model_trainer_logistic.max_seq_length


def test_transform_rnn(model_trainer_rnn):
    model_trainer_rnn.fit_transform(preprocessed_texts, labels)
    new_features = model_trainer_rnn.transform(["Deep learning is great"])
    assert new_features.shape[0] == 1
    assert new_features.shape[1] == model_trainer_rnn.max_seq_length


def test_evaluate_logistic(model_trainer_logistic):
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_texts, labels, test_size=0.5, random_state=42, stratify=labels
    )
    model_trainer_logistic.fit_transform(X_train, y_train)
    evaluation = model_trainer_logistic.evaluate(X_test, y_test)
    assert "accuracy" in evaluation


def test_evaluate_rnn(model_trainer_rnn):
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_texts, labels, test_size=0.5, random_state=42, stratify=labels
    )
    model_trainer_rnn.fit_transform(X_train, y_train)
    evaluation = model_trainer_rnn.evaluate(X_test, y_test)
    assert "accuracy" in evaluation


def test_predict_logistic(model_trainer_logistic):
    model_trainer_logistic.fit_transform(preprocessed_texts, labels)
    predictions = model_trainer_logistic.predict(["Deep learning is great"])
    assert predictions is not None


def test_predict_rnn(model_trainer_rnn):
    model_trainer_rnn.fit_transform(preprocessed_texts, labels)
    predictions = model_trainer_rnn.predict(["Deep learning is great"])
    assert predictions is not None
