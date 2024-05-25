import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.reddit_api_interaction import fetch_reddit_data


@pytest.fixture
def mock_reddit_api_keys(mocker):
    return mocker.patch(
        "src.reddit_api_interaction.get_reddit_api_keys",
        return_value={
            "client_id": "fake_client_id",
            "client_secret": "fake_client_secret",
            "user_agent": "fake_user_agent",
        },
    )


@pytest.fixture
def mock_config_path(mocker, tmp_path):
    config_file = tmp_path / "config.ini"
    mock = mocker.patch(
        "src.reddit_api_interaction.get_config_path", return_value=str(config_file)
    )
    return mock


@pytest.fixture
def mock_praw(mocker):
    mock_reddit = mocker.patch("praw.Reddit").return_value
    mock_subreddit = mock_reddit.subreddit.return_value
    mock_submission = Mock()
    mock_submission.title = "Test Title"
    mock_submission.score = 100
    mock_submission.selftext = "Test Content"
    mock_submission.url = "http://example.com"
    mock_submission.num_comments = 10
    mock_submission.created = 1610000000.0
    mock_submission.id = "test_id"
    mock_subreddit.hot.return_value = [mock_submission]
    return mock_reddit


def test_fetch_reddit_data(mock_reddit_api_keys, mock_config_path, mock_praw, tmp_path):
    # sample params
    subreddits = ["test_subreddit"]
    num_posts = 1

    expected_output = [
        {
            "title": "Test Title",
            "score": 100,
            "content": "Test Content",
            "url": "http://example.com",
            "num_comments": 10,
            "created": 1610000000.0,
            "id": "test_id",
        }
    ]

    result = fetch_reddit_data(subreddits, num_posts)

    assert result == expected_output

    # verify data has been saved to a test dir
    config_dir = tmp_path
    csv_filename = config_dir / "reddit_data/test_subreddit_data.csv"

    # check file has been created correctly
    assert csv_filename.exists()

    # cleanup
    os.remove(csv_filename)
