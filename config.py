"""
Configuration management for Reddit Persona Generator.

This module handles all configuration settings including API credentials,
model parameters, and application settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class for the application."""
    
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'PersonaGenerator/1.0')
    
    # LLM Configuration
    DEFAULT_MODEL_NAME = "facebook/opt-125m"
    MAX_RESPONSE_LENGTH = 200
    MAX_QUOTE_LENGTH = 100
    
    # NLP Configuration
    SPACY_MODEL = "en_core_web_sm"
    MAX_POSTS_TO_ANALYZE = 50
    MAX_COMMENTS_TO_ANALYZE = 100
    
    # Flask Configuration
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    FLASK_DEBUG = False
    
    # Output Configuration
    OUTPUT_DIR = "output"
    
    # Quantization Configuration
    QUANTIZATION_CONFIG = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    
    # LLM Generation Parameters
    LLM_GENERATION_CONFIG = {
        "temperature": 0.5,  # Further reduced for more focused, coherent responses
        "do_sample": True,
        "top_p": 0.75,  # Further reduced for more focused token selection
        "top_k": 20,  # Further reduced for more focused selection
        "repetition_penalty": 1.5,  # Increased further to reduce repetition
        "use_cache": True,
        "no_repeat_ngram_size": 5,  # Increased to prevent longer repetitions
        "pad_token_id": None  # Will be set during tokenization
    }
    
    # Quote Generation Parameters
    QUOTE_GENERATION_CONFIG = {
        "temperature": 0.6,  # Further reduced for more coherent quotes
        "do_sample": True,
        "top_p": 0.7,  # Further reduced for more focused quotes
        "top_k": 15,  # Further reduced for more focused quotes
        "repetition_penalty": 1.6,  # Increased further to prevent repetitive quotes
        "use_cache": True,
        "no_repeat_ngram_size": 5,  # Increased to prevent repetitions
        "pad_token_id": None  # Will be set during tokenization
    }
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate that all required configuration is present."""
        required_configs = [
            'REDDIT_CLIENT_ID',
            'REDDIT_CLIENT_SECRET'
        ]
        
        missing_configs = []
        for config in required_configs:
            if not getattr(cls, config):
                missing_configs.append(config)
        
        if missing_configs:
            raise ValueError(f"Missing required configuration: {', '.join(missing_configs)}")
    
    @classmethod
    def get_reddit_config(cls) -> Dict[str, str]:
        """Get Reddit API configuration."""
        return {
            'client_id': cls.REDDIT_CLIENT_ID,
            'client_secret': cls.REDDIT_CLIENT_SECRET,
            'user_agent': cls.REDDIT_USER_AGENT
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            'model_name': cls.DEFAULT_MODEL_NAME,
            'max_response_length': cls.MAX_RESPONSE_LENGTH,
            'max_quote_length': cls.MAX_QUOTE_LENGTH,
            'quantization_config': cls.QUANTIZATION_CONFIG,
            'generation_config': cls.LLM_GENERATION_CONFIG,
            'quote_generation_config': cls.QUOTE_GENERATION_CONFIG
        }
    
    @classmethod
    def get_nlp_config(cls) -> Dict[str, Any]:
        """Get NLP configuration."""
        return {
            'spacy_model': cls.SPACY_MODEL,
            'max_posts': cls.MAX_POSTS_TO_ANALYZE,
            'max_comments': cls.MAX_COMMENTS_TO_ANALYZE
        }
    
    @classmethod
    def get_flask_config(cls) -> Dict[str, Any]:
        """Get Flask configuration."""
        return {
            'host': cls.FLASK_HOST,
            'port': cls.FLASK_PORT,
            'debug': cls.FLASK_DEBUG
        } 