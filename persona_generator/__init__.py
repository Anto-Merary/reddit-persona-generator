"""
Reddit Persona Generator Package

A sophisticated AI-powered system for generating comprehensive user personas
from Reddit profiles using quantized LLMs and advanced NLP techniques.
"""

from .core import PersonaGenerator
from .reddit_client import RedditClient
from .nlp_processor import NLPProcessor
from .llm_handler import LLMHandler
from .persona_builder import PersonaBuilder

__version__ = "1.0.0"
__author__ = "AI/LLM Engineer"

__all__ = [
    "PersonaGenerator",
    "RedditClient", 
    "NLPProcessor",
    "LLMHandler",
    "PersonaBuilder"
] 