import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.feature_extraction import FeatureExtractionMethod, FeatureExtractor

texts = ["I love machine learning", "Natural Language Processing is amazing"]
texts_new = ["Deep learning is part of machine learning"]


@pytest.fixture
def sample_texts():
    return texts


@pytest.fixture
def new_texts():
    return texts_new


def test_initialization_bow():
    extractor = FeatureExtractor(FeatureExtractionMethod.BOW, max_features=1000)
    assert extractor.method == FeatureExtractionMethod.BOW
    assert extractor.max_features == 1000


def test_initialization_tfidf():
    extractor = FeatureExtractor(FeatureExtractionMethod.TFIDF, max_features=1000)
    assert extractor.method == FeatureExtractionMethod.TFIDF
    assert extractor.max_features == 1000


def test_initialization_glove():
    extractor = FeatureExtractor(FeatureExtractionMethod.GLOVE)
    assert extractor.method == FeatureExtractionMethod.GLOVE
    assert extractor.max_features is None


def test_fit_transform_bow(sample_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.BOW, max_features=1000)
    features = extractor.fit_transform(sample_texts)
    assert isinstance(features, csr_matrix)
    assert features.shape == (2, features.shape[1])


def test_fit_transform_tfidf(sample_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.TFIDF, max_features=1000)
    features = extractor.fit_transform(sample_texts)
    assert isinstance(features, csr_matrix)
    assert features.shape == (2, features.shape[1])


def test_fit_transform_glove(sample_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.GLOVE)
    features = extractor.fit_transform(sample_texts)
    assert isinstance(features, np.ndarray)
    assert features.shape == (
        2,
        extractor.nlp("I love machine learning").vector.shape[0],
    )


def test_transform_bow(sample_texts, new_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.BOW, max_features=1000)
    extractor.fit_transform(sample_texts)
    new_features = extractor.transform(new_texts)
    assert isinstance(new_features, csr_matrix)
    assert new_features.shape == (1, new_features.shape[1])


def test_transform_tfidf(sample_texts, new_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.TFIDF, max_features=1000)
    extractor.fit_transform(sample_texts)
    new_features = extractor.transform(new_texts)
    assert isinstance(new_features, csr_matrix)
    assert new_features.shape == (1, new_features.shape[1])


def test_transform_glove(sample_texts, new_texts):
    extractor = FeatureExtractor(FeatureExtractionMethod.GLOVE)
    extractor.fit_transform(sample_texts)
    new_features = extractor.transform(new_texts)
    assert isinstance(new_features, np.ndarray)
    assert new_features.shape == (
        1,
        extractor.nlp("Deep learning is part of machine learning").vector.shape[0],
    )
