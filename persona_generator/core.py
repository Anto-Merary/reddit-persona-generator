"""
Core module for the Reddit Persona Generator.

This module provides the main PersonaGenerator class that orchestrates
all components to generate comprehensive user personas from Reddit profiles.
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from config import Config
from .utils import (
    setup_logging, extract_username_from_url, ensure_output_directory,
    validate_user_data, clean_text
)
from .reddit_client import RedditClient
from .nlp_processor import NLPProcessor
from .llm_handler import LLMHandler
from .persona_builder import PersonaBuilder


class PersonaGenerator:
    """
    Main class for generating comprehensive personas from Reddit user profiles.
    
    This class orchestrates all components including Reddit data fetching,
    NLP processing, LLM generation, and persona building.
    """
    
    def __init__(self, reddit_client_id: Optional[str] = None, 
                 reddit_client_secret: Optional[str] = None,
                 reddit_user_agent: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize the Persona Generator.
        
        Args:
            reddit_client_id: Reddit API client ID (optional, uses config)
            reddit_client_secret: Reddit API client secret (optional, uses config)
            reddit_user_agent: Reddit API user agent (optional, uses config)
            model_name: LLM model name (optional, uses config)
        """
        # Setup logging
        self.logger = setup_logging()
        
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.reddit_client = self._initialize_reddit_client(
            reddit_client_id, reddit_client_secret, reddit_user_agent
        )
        self.nlp_processor = NLPProcessor()
        self.llm_handler = LLMHandler(model_name)
        self.persona_builder = PersonaBuilder(
            self.llm_handler, self.nlp_processor, self.reddit_client
        )
        
        # Ensure output directory exists
        ensure_output_directory(Config.OUTPUT_DIR)
        
        self.logger.info("PersonaGenerator initialized successfully")
    
    def _initialize_reddit_client(self, client_id: Optional[str], 
                                 client_secret: Optional[str], 
                                 user_agent: Optional[str]) -> RedditClient:
        """Initialize Reddit client with provided or config credentials."""
        if client_id and client_secret:
            return RedditClient(client_id, client_secret, user_agent or Config.REDDIT_USER_AGENT)
        else:
            return RedditClient.from_config()
    
    def generate_persona_from_url(self, profile_url: str, output_file: Optional[str] = None) -> Dict:
        """
        Generate a persona from a Reddit profile URL.
        
        Args:
            profile_url: Reddit profile URL
            output_file: Optional output file path
            
        Returns:
            Dictionary containing persona data and metadata
        """
        try:
            # Extract username from URL
            username = extract_username_from_url(profile_url)
            self.logger.info(f"Generating persona for user: {username}")
            
            # Fetch user data
            user_data = self.reddit_client.fetch_user_data(username)
            
            # Validate user data
            if not validate_user_data(user_data):
                return {
                    'error': f"Invalid user data for '{username}'",
                    'username': username,
                    'output_file': None
                }
            
            # Check if user has content
            if not user_data['posts'] and not user_data['comments']:
                return {
                    'error': f"User '{username}' has no public posts or comments available for analysis.",
                    'username': username,
                    'output_file': None
                }
            
            # Generate persona
            return self.generate_persona_from_data(user_data, output_file)
            
        except Exception as e:
            error_msg = f"Error generating persona from URL: {e}"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'username': 'unknown',
                'output_file': None
            }
    
    def generate_persona_from_data(self, user_data: Dict, output_file: Optional[str] = None) -> Dict:
        """
        Generate a persona from user data.
        
        Args:
            user_data: Reddit user data dictionary
            output_file: Optional output file path
            
        Returns:
            Dictionary containing persona data and metadata
        """
        try:
            username = user_data['username']
            self.logger.info(f"Processing user data for: {username}")
            
            # Clean and prepare text data
            all_texts = self._prepare_text_data(user_data)
            
            # Extract themes using NLP
            self.logger.info("Extracting themes and characteristics...")
            themes = self.nlp_processor.extract_themes_and_entities(all_texts)
            
            # Build persona
            self.logger.info("Building comprehensive persona...")
            persona = self.persona_builder.build_persona(user_data, themes)
            
            # Create output file if requested
            if output_file or Config.OUTPUT_DIR:
                if not output_file:
                    output_file = f"{Config.OUTPUT_DIR}/persona_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                # Write formatted report
                report_content = self.persona_builder.format_persona_report(
                    username, persona, user_data, themes
                )
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                self.logger.info(f"Report saved to: {output_file}")
            
            return {
                'username': username,
                'persona': persona,
                'output_file': output_file,
                'user_stats': {
                    'posts': len(user_data['posts']),
                    'comments': len(user_data['comments']),
                    'comment_karma': user_data['comment_karma'],
                    'link_karma': user_data['link_karma']
                },
                'themes': themes
            }
            
        except Exception as e:
            error_msg = f"Error generating persona from data: {e}"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'username': user_data.get('username', 'unknown'),
                'output_file': None
            }
    
    def _prepare_text_data(self, user_data: Dict) -> list:
        """Prepare and clean text data for analysis."""
        all_texts = []
        
        # Clean and collect post content
        for post in user_data['posts']:
            if post['title']:
                all_texts.append(clean_text(post['title']))
            if post['body']:
                all_texts.append(clean_text(post['body']))
        
        # Clean and collect comment content
        for comment in user_data['comments']:
            if comment['body']:
                all_texts.append(clean_text(comment['body']))
        
        # Filter out empty texts
        return [text for text in all_texts if text.strip()]
    
    def get_system_info(self) -> Dict:
        """Get information about the system components."""
        return {
            'reddit_client': {
                'status': 'initialized',
                'max_posts': Config.MAX_POSTS_TO_ANALYZE,
                'max_comments': Config.MAX_COMMENTS_TO_ANALYZE
            },
            'nlp_processor': {
                'spacy_available': self.nlp_processor.nlp is not None,
                'spacy_model': Config.SPACY_MODEL
            },
            'llm_handler': self.llm_handler.get_model_info(),
            'config': {
                'output_dir': Config.OUTPUT_DIR,
                'model_name': Config.DEFAULT_MODEL_NAME
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm_handler'):
            self.llm_handler.cleanup()
        self.logger.info("PersonaGenerator cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Legacy compatibility class
class QuantizedLLMPersonaGenerator(PersonaGenerator):
    """
    Legacy class name for backward compatibility.
    
    This class provides the same functionality as PersonaGenerator
    but maintains the original class name for existing code.
    """
    
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, 
                 reddit_user_agent: str, model_name: str = "facebook/opt-125m"):
        """
        Initialize with legacy parameter structure.
        
        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
            model_name: LLM model name
        """
        super().__init__(reddit_client_id, reddit_client_secret, reddit_user_agent, model_name)
    
    def extract_username_from_url(self, profile_url: str) -> str:
        """Legacy method for extracting username from URL."""
        return extract_username_from_url(profile_url)
    
    def fetch_user_data(self, username: str) -> Dict:
        """Legacy method for fetching user data."""
        return self.reddit_client.fetch_user_data(username)
    
    def clean_text(self, text: str) -> str:
        """Legacy method for cleaning text."""
        return clean_text(text)
    
    def extract_themes_with_spacy(self, texts: list) -> Dict:
        """Legacy method for extracting themes."""
        return self.nlp_processor.extract_themes_and_entities(texts)
    
    def generate_persona_with_quantized_llm(self, user_data: Dict, themes: Dict) -> Dict:
        """Legacy method for generating persona."""
        return self.persona_builder.build_persona(user_data, themes)
    
    def generate_persona_report(self, profile_url: str, output_file: Optional[str] = None) -> Dict:
        """Legacy method for generating persona report."""
        return self.generate_persona_from_url(profile_url, output_file) 