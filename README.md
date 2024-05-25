# Reddit-Api-Script

## Set up Reddit API

1. Go to https://www.reddit.com/prefs/apps and create an application for your script.
2. Create a `config.ini` file copying the configuration from the `config_template.ini` file and substitute the xxx with the actual keys from the Reddit API.

## Exploring Subreddits

You can get posts from different subreddits by running this line:
```commandline
python src/reddit_api_interaction.py
```
This will put the data from the subreddits in the `reddit_data` directory.

The default subreddit is r/nursing but you can specify more in the subreddits list in the `reddit_api_interaction` file.

## Environment Setup
1. For proper functioning of the modules, make sure to create a `.env` file in the root directory of the project.
2. Copy the configuration in the `.env.example` file, and substitute with the values for your computer.

Note if you are still having some kind of issue running the modules (errors similar to module does not exist), run  ```export PYTHONPATH=$(pwd)``` and it should do the trick.
