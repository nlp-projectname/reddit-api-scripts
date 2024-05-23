# Reddit-Api-Script

## Set up Reddit API

1. Go to https://www.reddit.com/prefs/apps and create an application for your script.
2. Create a config.ini file copying the configuration from the config_template.ini file and substitute the xxx with the actual keys from the Reddit API.

## Exploring Subreddits

You can get posts from different subreddits by running this line:
```commandline
python3 -m src.reddit_api_interaction
```
This will put the data from the subreddits in the `reddit_data` directory.

The default subreddit is r/nursing but you can specify more in the subreddits list in the `reddit_api_interaction` file.