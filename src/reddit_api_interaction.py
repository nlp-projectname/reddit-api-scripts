import praw
import os
import pandas as pd
from typing import List, Dict
from src.utils import get_reddit_api_keys, get_config_path


def fetch_reddit_data(subreddits, num_posts):
    config_path = get_config_path()
    api_keys = get_reddit_api_keys(config_path)

    reddit = praw.Reddit(client_id=api_keys['client_id'],
                         client_secret=api_keys['client_secret'],
                         user_agent=api_keys['user_agent'])

    def get_subreddit_data(subreddit_name):
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for submission in subreddit.hot(limit=num_posts):
            posts.append({
                'title': submission.title,
                'score': submission.score,
                'content': submission.selftext,
                'url': submission.url,
                'num_comments': submission.num_comments,
                'created': submission.created,
                'id': submission.id
            })
        return posts

    config_dir = os.path.dirname(config_path)
    all_data = []

    for subreddit in subreddits:
        subreddit_data = get_subreddit_data(subreddit)
        df = pd.DataFrame(subreddit_data)
        csv_dir = os.path.join(config_dir, 'reddit_data')
        os.makedirs(csv_dir, exist_ok=True)  # ensure the directory exists
        csv_filename = os.path.join(csv_dir, f'{subreddit}_data.csv')
        # print(f"Saving CSV to: {csv_filename}")
        df.to_csv(csv_filename, index=False)
        all_data.extend(subreddit_data)

    return all_data


if __name__ == '__main__':
    # Define subreddits and number of posts
    subreddits = ['nursing', 'Teachers']
    num_posts = 1000
    
    # Fetch Reddit data
    all_data = fetch_reddit_data(subreddits, num_posts)

    print(all_data)
