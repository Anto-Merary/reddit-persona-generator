"""
Utility functions for the Reddit Persona Generator.

This module contains common helper functions used across the application.
"""

import os
import re
import logging
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('persona_generator.log')
        ]
    )
    return logging.getLogger(__name__)


def extract_username_from_url(profile_url: str) -> str:
    """
    Extract username from Reddit profile URL.
    
    Args:
        profile_url: Reddit profile URL
        
    Returns:
        Extracted username
        
    Raises:
        ValueError: If URL format is invalid
    """
    try:
        parsed = urlparse(profile_url)
        if 'reddit.com' not in parsed.netloc:
            raise ValueError("Not a valid Reddit URL")
        
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'user':
            return path_parts[1]
        else:
            raise ValueError("Invalid Reddit profile URL format")
            
    except Exception as e:
        raise ValueError(f"Error parsing URL: {e}")


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, special characters, and Reddit formatting.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text suitable for NLP analysis
    """
    if not text:
        return ""
    
    # Remove URLs (http/https)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove markdown links
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
    text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Remove strikethrough
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove inline code
    text = re.sub(r'^>', '', text, flags=re.MULTILINE)  # Remove quote markers
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_timestamp(timestamp: float) -> str:
    """
    Format Unix timestamp to readable date.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted date string
    """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')


def format_keywords(keywords: List) -> str:
    """
    Format keywords list, handling both tuple and string formats.
    
    Args:
        keywords: List of keywords (can be tuples or strings)
        
    Returns:
        Formatted keywords string
    """
    if not keywords:
        return "None"
    
    if keywords and isinstance(keywords[0], tuple):
        return ', '.join([word for word, _ in keywords])
    else:
        return ', '.join([str(word) for word in keywords])


def ensure_output_directory(output_dir: str = "output") -> None:
    """
    Ensure output directory exists.
    
    Args:
        output_dir: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)


def get_account_age(created_utc: float) -> str:
    """
    Calculate and categorize account age.
    
    Args:
        created_utc: Account creation timestamp
        
    Returns:
        Formatted account age string
    """
    account_age = datetime.now() - datetime.fromtimestamp(created_utc)
    years = account_age.days // 365
    months = (account_age.days % 365) // 30
    
    if years >= 5:
        return f"{years} years (veteran user)"
    elif years >= 2:
        return f"{years} years (established user)"
    elif years >= 1:
        return f"{years} year {months} months (regular user)"
    else:
        return f"{months} months (newer user)"


def analyze_activity_level(user_data: Dict) -> str:
    """
    Analyze user's activity level and patterns.
    
    Args:
        user_data: User data dictionary
        
    Returns:
        Activity level description
    """
    total_posts = len(user_data['posts'])
    total_comments = len(user_data['comments'])
    account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
    
    posts_per_day = total_posts / max(account_age_days, 1)
    comments_per_day = total_comments / max(account_age_days, 1)
    
    if posts_per_day > 1 or comments_per_day > 5:
        return "Very active (posts/comments frequently)"
    elif posts_per_day > 0.1 or comments_per_day > 1:
        return "Moderately active (regular participation)"
    else:
        return "Low activity (occasional participation)"


def analyze_engagement_style(user_data: Dict) -> str:
    """
    Analyze how the user engages with Reddit.
    
    Args:
        user_data: User data dictionary
        
    Returns:
        Engagement style description
    """
    posts = len(user_data['posts'])
    comments = len(user_data['comments'])
    
    if posts == 0 and comments == 0:
        return "No recent activity"
    elif posts == 0:
        return "Comment-only user (never posts)"
    elif comments == 0:
        return "Post-only user (never comments)"
    else:
        ratio = comments / max(posts, 1)
        if ratio > 10:
            return "Heavy commenter (prefers discussion)"
        elif ratio > 3:
            return "Balanced participant (posts and comments)"
        else:
            return "Content creator (prefers posting)"


def get_relevant_citations(description: str, user_data: Dict, max_citations: int = 3) -> List[str]:
    """
    Find relevant citations (URLs) for a persona description.
    
    Args:
        description: Generated persona description
        user_data: User data containing posts and comments
        max_citations: Maximum number of citations to return
        
    Returns:
        List of relevant URLs
    """
    citations = []
    
    # Simple keyword matching for citations
    description_lower = description.lower()
    description_words = [word for word in description_lower.split() if len(word) > 3]
    
    # Check posts for relevance
    for post in user_data['posts'][:10]:  # Check first 10 posts
        post_text = f"{post['title']} {post.get('body', '')}".lower()
        if any(word in post_text for word in description_words):
            citations.append(post['url'])
    
    # Check comments for relevance
    for comment in user_data['comments'][:10]:  # Check first 10 comments
        comment_text = comment['body'].lower()
        if any(word in comment_text for word in description_words):
            citations.append(comment['url'])
    
    return citations[:max_citations]


def validate_user_data(user_data: Dict) -> bool:
    """
    Validate that user data contains required fields.
    
    Args:
        user_data: User data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['username', 'created_utc', 'posts', 'comments']
    return all(field in user_data for field in required_fields)


def setup_protobuf_compatibility() -> None:
    """Setup protobuf compatibility for the application."""
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    try:
        import google.protobuf
        # Force downgrade behavior for compatibility
        if hasattr(google.protobuf, 'Message'):
            google.protobuf.Message._CheckCalledFromGeneratedFile = lambda: None
    except (ImportError, AttributeError):
        # Protobuf not installed or doesn't have Message attribute
        pass 