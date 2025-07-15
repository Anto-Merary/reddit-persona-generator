"""
Reddit API client for the Persona Generator.

This module handles all interactions with the Reddit API using PRAW.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import praw
from config import Config


class RedditClient:
    """Client for interacting with Reddit API."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize the Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        self.logger.info("Reddit client initialized successfully")
    
    @classmethod
    def from_config(cls) -> 'RedditClient':
        """
        Create RedditClient instance from configuration.
        
        Returns:
            RedditClient instance
        """
        reddit_config = Config.get_reddit_config()
        return cls(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config['user_agent']
        )
    
    def fetch_user_data(self, username: str) -> Dict:
        """
        Fetch user posts and comments from Reddit with comprehensive data.
        
        Args:
            username: Reddit username
            
        Returns:
            Dictionary containing user data, posts, and comments with URLs
            
        Raises:
            ValueError: If user is not found or profile is private
        """
        try:
            self.logger.info(f"Fetching data for user: {username}")
            
            # Get Reddit user object
            redditor = self.reddit.redditor(username)
            
            # Check if user exists and is accessible
            try:
                _ = redditor.created_utc
            except Exception:
                raise ValueError(f"User '{username}' not found or profile is private")
            
            user_data = {
                'username': username,
                'created_utc': redditor.created_utc,
                'comment_karma': redditor.comment_karma,
                'link_karma': redditor.link_karma,
                'posts': [],
                'comments': []
            }
            
            # Fetch posts (submissions) with full URLs
            self.logger.info("Fetching user posts...")
            nlp_config = Config.get_nlp_config()
            max_posts = nlp_config['max_posts']
            
            for submission in redditor.submissions.new(limit=max_posts):
                post_data = {
                    'title': submission.title,
                    'body': submission.selftext,
                    'url': f"https://reddit.com{submission.permalink}",
                    'subreddit': submission.subreddit.display_name,
                    'score': submission.score,
                    'created_utc': submission.created_utc,
                    'num_comments': submission.num_comments,
                    'id': submission.id
                }
                user_data['posts'].append(post_data)
            
            # Fetch comments with full URLs
            self.logger.info("Fetching user comments...")
            max_comments = nlp_config['max_comments']
            
            for comment in redditor.comments.new(limit=max_comments):
                comment_data = {
                    'body': comment.body,
                    'url': f"https://reddit.com{comment.permalink}",
                    'subreddit': comment.subreddit.display_name,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'id': comment.id,
                    'parent_id': comment.parent_id
                }
                user_data['comments'].append(comment_data)
            
            self.logger.info(f"Fetched {len(user_data['posts'])} posts and {len(user_data['comments'])} comments")
            return user_data
            
        except Exception as e:
            self.logger.error(f"Error fetching user data: {e}")
            raise
    
    def get_user_subreddits(self, user_data: Dict) -> Dict[str, int]:
        """
        Get subreddits the user participates in with frequency.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Dictionary mapping subreddit names to participation counts
        """
        subreddit_counts = {}
        
        # Count posts per subreddit
        for post in user_data['posts']:
            sub = post['subreddit']
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
        
        # Count comments per subreddit
        for comment in user_data['comments']:
            sub = comment['subreddit']
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
        
        # Sort by frequency
        return dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_content(self, user_data: Dict, content_type: str = 'both', limit: int = 5) -> List[Dict]:
        """
        Get top content by score.
        
        Args:
            user_data: User data dictionary
            content_type: Type of content ('posts', 'comments', or 'both')
            limit: Maximum number of items to return
            
        Returns:
            List of top content items
        """
        top_content = []
        
        if content_type in ['posts', 'both']:
            for post in sorted(user_data['posts'], key=lambda x: x['score'], reverse=True)[:limit]:
                top_content.append({
                    'type': 'post',
                    'title': post['title'],
                    'body': post.get('body', ''),
                    'url': post['url'],
                    'subreddit': post['subreddit'],
                    'score': post['score'],
                    'created_utc': post['created_utc']
                })
        
        if content_type in ['comments', 'both']:
            for comment in sorted(user_data['comments'], key=lambda x: x['score'], reverse=True)[:limit]:
                top_content.append({
                    'type': 'comment',
                    'body': comment['body'],
                    'url': comment['url'],
                    'subreddit': comment['subreddit'],
                    'score': comment['score'],
                    'created_utc': comment['created_utc']
                })
        
        # Sort all content by score and return top items
        return sorted(top_content, key=lambda x: x['score'], reverse=True)[:limit]
    
    def analyze_posting_patterns(self, user_data: Dict) -> Dict:
        """
        Analyze user's posting patterns and timing.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Dictionary with posting pattern analysis
        """
        # Analyze posting frequency
        posts_with_times = [(post['created_utc'], 'post') for post in user_data['posts']]
        comments_with_times = [(comment['created_utc'], 'comment') for comment in user_data['comments']]
        
        all_activity = sorted(posts_with_times + comments_with_times)
        
        if not all_activity:
            return {
                'most_active_hour': 'No activity',
                'activity_pattern': 'No activity',
                'posting_frequency': 'No activity'
            }
        
        # Analyze posting times
        hours = [datetime.fromtimestamp(timestamp).hour for timestamp, _ in all_activity]
        
        # Find peak hours
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 12
        
        # Categorize activity pattern
        if 6 <= peak_hour <= 11:
            activity_pattern = 'Morning poster (6-11 AM)'
        elif 12 <= peak_hour <= 17:
            activity_pattern = 'Afternoon poster (12-5 PM)'
        elif 18 <= peak_hour <= 22:
            activity_pattern = 'Evening poster (6-10 PM)'
        else:
            activity_pattern = 'Night owl (11 PM-5 AM)'
        
        # Calculate posting frequency
        account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
        total_posts = len(user_data['posts'])
        total_comments = len(user_data['comments'])
        
        posts_per_day = total_posts / max(account_age_days, 1)
        comments_per_day = total_comments / max(account_age_days, 1)
        
        if posts_per_day > 1 or comments_per_day > 5:
            posting_frequency = "Very active (posts/comments frequently)"
        elif posts_per_day > 0.1 or comments_per_day > 1:
            posting_frequency = "Moderately active (regular participation)"
        else:
            posting_frequency = "Low activity (occasional participation)"
        
        return {
            'most_active_hour': peak_hour,
            'activity_pattern': activity_pattern,
            'posting_frequency': posting_frequency,
            'posts_per_day': posts_per_day,
            'comments_per_day': comments_per_day
        }
    
    def get_interaction_style(self, user_data: Dict) -> str:
        """
        Analyze how the user interacts with others.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Interaction style description
        """
        # Count replies vs original content
        reply_count = 0
        for comment in user_data['comments']:
            if comment['parent_id'] and not comment['parent_id'].startswith('t3_'):
                reply_count += 1
        
        total_comments = len(user_data['comments'])
        
        if total_comments == 0:
            return 'No interaction data'
        
        reply_ratio = reply_count / total_comments
        
        if reply_ratio > 0.7:
            return 'Conversational (frequently replies to others)'
        elif reply_ratio > 0.3:
            return 'Balanced (mix of original and replies)'
        else:
            return 'Independent (prefers original comments)' 