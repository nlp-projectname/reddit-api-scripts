import pandas as pd
import pytest

from src.text_preprocessing import TextPreprocessor
from src.utils import contractions_dict


@pytest.fixture
def sample_text():
    return "I'm happy because I can't go to the park. I am happy 😊"


@pytest.fixture
def expected_expanded_text():
    return "i am happy because i cannot go to the park. i am happy 😊"


@pytest.fixture
def expected_emoji_text():
    return "i am happy because i cannot go to the park. i am happy smiling face with smiling eyes"


@pytest.fixture
def expected_preprocessed_text():
    return "happi go park happi smile face smile eye"


@pytest.fixture
def sample_dataframe():
    data = {
        "title": ["Sample Title"],
        "score": [1],
        "content": ["I'm happy because I can't go to the park. I am happy 😊"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def expected_processed_dataframe():
    return pd.DataFrame(
        {
            "title": ["Sample Title"],
            "score": [1],
            "content": ["happi go park happi smile face smile eye"],
        }
    )


def test_expand_contractions(sample_text, expected_expanded_text):
    preprocessor = TextPreprocessor()
    assert (
        preprocessor._expand_contractions(sample_text.lower()) == expected_expanded_text
    )


def test_convert_emojis(expected_expanded_text, expected_emoji_text):
    preprocessor = TextPreprocessor()
    assert preprocessor._convert_emojis(expected_expanded_text) == expected_emoji_text


def test_preprocess_text(sample_text, expected_preprocessed_text):
    preprocessor = TextPreprocessor()
    assert preprocessor.preprocess_text(sample_text) == expected_preprocessed_text


def test_preprocess_dataframe(sample_dataframe, expected_processed_dataframe):
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(sample_dataframe.copy())
    pd.testing.assert_frame_equal(processed_df, expected_processed_dataframe)


def test_preprocess_dataframe_multiple_rows():
    data = {
        "title": ["Title 1", "Title 2"],
        "score": [1, 2],
        "content": ["I'm happy 😊", "I can't go to the park 😢"],
    }
    df = pd.DataFrame(data)
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(df.copy())
    expected_contents = [
        "happi smile face smile eye",
        "go park cri face",
    ]  # hardcoded (maybe better to change to reflect script)
    assert processed_df["content"].iloc[0] == expected_contents[0]
    assert processed_df["content"].iloc[1] == expected_contents[1]
