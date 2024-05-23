import configparser
import os

from typing import Dict, Any


def get_config_path() -> str:
    """Get the path to the config.ini file located at the root of the project."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')


def get_reddit_api_keys(file_path: str) -> Dict[str, Any]:
    config = configparser.ConfigParser()
    config.read(file_path)
    return config['REDDIT_API']  # type: ignore