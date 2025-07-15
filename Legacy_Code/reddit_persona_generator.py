#!/usr/bin/env python3
"""
Reddit Persona Generator with Quantized LLM and Flask API

This script analyzes Reddit user profiles to generate comprehensive personas
using a quantized LLM (facebook/opt-125m) with 4-bit quantization. It provides
both a Flask API endpoint and direct script execution capabilities.

Features:
- Reddit API integration using PRAW
- Text cleaning and preprocessing
- NLP analysis with spaCy
- Quantized LLM persona generation
- Flask API with web interface support
- Comprehensive error handling and edge case management

Author: Anto Merary S
Date: 2025
"""

import os
import re
import logging
from typing import Dict, List
from urllib.parse import urlparse
from datetime import datetime
import praw
import spacy
import nltk
from textblob import TextBlob
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the new modular components for backward compatibility
from persona_generator.core import PersonaGenerator, QuantizedLLMPersonaGenerator

# Try to import BitsAndBytesConfig with fallback
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    # Fallback for older transformers versions
    print("⚠️  BitsAndBytesConfig not available in this transformers version")
    print("   Quantization will be disabled. For quantization support:")
    print("   pip install transformers>=4.20.0 bitsandbytes>=0.35.0")
    BitsAndBytesConfig = None
    QUANTIZATION_AVAILABLE = False
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Fix protobuf compatibility issues
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Additional protobuf fix
try:
    import google.protobuf
    # Force downgrade behavior for compatibility
    if hasattr(google.protobuf, 'Message'):
        google.protobuf.Message._CheckCalledFromGeneratedFile = lambda: None
except (ImportError, AttributeError):
    # Protobuf not installed or doesn't have Message attribute, will be handled later
    pass


class QuantizedLLMPersonaGenerator:
    """
    A class to generate comprehensive personas from Reddit user profiles
    using a quantized LLM and NLP techniques.
    """
    
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, 
                 reddit_user_agent: str, model_name: str = "facebook/opt-125m"):
        """
        Initialize the Reddit Persona Generator with quantized LLM.
        
        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
            model_name: Hugging Face model name for LLM
        """
        # Setup logging first (needed by other methods)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Initialize quantized LLM
        self._setup_quantized_llm(model_name)
        
        # Initialize NLP components
        self._setup_nlp()
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
    
    def _setup_quantized_llm(self, model_name: str):
        """
        Initialize the quantized LLM model with 4-bit quantization.
        Optimized for GPU usage with protobuf error handling.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.logger.info(f"Loading quantized model: {model_name}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.logger.warning("No GPU detected, will use CPU")
            
            # Load tokenizer first with error handling
            try:
                self.logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.logger.info("✓ Tokenizer loaded successfully")
            except Exception as tokenizer_error:
                self.logger.error(f"Error loading tokenizer: {tokenizer_error}")
                if "protobuf" in str(tokenizer_error).lower() or "messagefactory" in str(tokenizer_error).lower():
                    self.logger.error("Protobuf compatibility issue detected. Please run:")
                    self.logger.error("pip install protobuf==3.20.3")
                    self.logger.error("pip install transformers==4.24.0")
                raise
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load with 4-bit quantization first (optimal for RTX Cards)
            if torch.cuda.is_available():
                # Check if quantization is available
                if QUANTIZATION_AVAILABLE:
                    try:
                        self.logger.info("Attempting to load model with 4-bit quantization on GPU...")
                        
                        # Configure 4-bit quantization optimized for RTX 4060
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        
                        # Load model with quantization
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=quantization_config,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        
                        self.logger.info("✓ Model loaded successfully with 4-bit quantization on GPU")
                        self.device = "cuda"
                        
                    except Exception as quant_error:
                        if "protobuf" in str(quant_error).lower() or "messagefactory" in str(quant_error).lower():
                            self.logger.error("Protobuf compatibility issue during model loading!")
                            self.logger.error("Please run: pip install -r requirements_fixed.txt")
                            raise
                        
                        self.logger.warning(f"4-bit quantization failed: {quant_error}")
                        self.logger.info("Trying regular GPU loading...")
                        
                        # Fallback to regular GPU loading
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True
                            )
                            
                            self.logger.info("✓ Model loaded successfully on GPU without quantization")
                            self.device = "cuda"
                            
                        except Exception as gpu_error:
                            if "protobuf" in str(gpu_error).lower() or "messagefactory" in str(gpu_error).lower():
                                self.logger.error("Protobuf compatibility issue during GPU model loading!")
                                self.logger.error("Please run: pip install -r requirements_fixed.txt")
                                raise
                                
                            self.logger.warning(f"GPU loading failed: {gpu_error}")
                            self.logger.info("Falling back to CPU loading...")
                            
                            # Final fallback to CPU
                            try:
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_name,
                                    torch_dtype=torch.float32,
                                    trust_remote_code=True
                                )
                                
                                self.logger.info("✓ Model loaded successfully on CPU")
                                self.device = "cpu"
                                
                            except Exception as cpu_error:
                                if "protobuf" in str(cpu_error).lower() or "messagefactory" in str(cpu_error).lower():
                                    self.logger.error("Protobuf compatibility issue during CPU model loading!")
                                    self.logger.error("Please run the following commands to fix:")
                                    self.logger.error("pip uninstall protobuf -y")
                                    self.logger.error("pip install protobuf==3.20.3")
                                    self.logger.error("pip install transformers==4.24.0")
                                raise
                else:
                    # Quantization not available, skip to regular GPU loading
                    self.logger.info("Quantization not available, loading model on GPU without quantization...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        
                        self.logger.info("✓ Model loaded successfully on GPU without quantization")
                        self.device = "cuda"
                        
                    except Exception as gpu_error:
                        if "protobuf" in str(gpu_error).lower() or "messagefactory" in str(gpu_error).lower():
                            self.logger.error("Protobuf compatibility issue during GPU model loading!")
                            self.logger.error("Please run: pip install -r requirements_fixed.txt")
                            raise
                            
                        self.logger.warning(f"GPU loading failed: {gpu_error}")
                        self.logger.info("Falling back to CPU loading...")
                        
                        # Final fallback to CPU
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32,
                                trust_remote_code=True
                            )
                            
                            self.logger.info("✓ Model loaded successfully on CPU")
                            self.device = "cpu"
                            
                        except Exception as cpu_error:
                            if "protobuf" in str(cpu_error).lower() or "messagefactory" in str(cpu_error).lower():
                                self.logger.error("Protobuf compatibility issue during CPU model loading!")
                                self.logger.error("Please run the following commands to fix:")
                                self.logger.error("pip uninstall protobuf -y")
                                self.logger.error("pip install protobuf==3.20.3")
                                self.logger.error("pip install transformers==4.24.0")
                            raise
            else:
                # No GPU available, load on CPU
                self.logger.info("Loading model on CPU...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    self.logger.info("✓ Model loaded successfully on CPU")
                    self.device = "cpu"
                    
                except Exception as cpu_error:
                    if "protobuf" in str(cpu_error).lower() or "messagefactory" in str(cpu_error).lower():
                        self.logger.error("Protobuf compatibility issue during CPU model loading!")
                        self.logger.error("Please run: pip install -r requirements_fixed.txt")
                    raise
            
        except Exception as e:
            self.logger.error(f"Error loading LLM: {e}")
            raise
    
    def _setup_nlp(self):
        """Initialize NLP components including spaCy and NLTK."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load spaCy model with better error handling
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
                import subprocess
                try:
                    # Try to download the model
                    result = subprocess.run(
                        ["python", "-m", "spacy", "download", "en_core_web_sm"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    self.logger.info("spaCy model downloaded successfully")
                    self.nlp = spacy.load("en_core_web_sm")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to download spaCy model: {e}")
                    self.logger.info("Falling back to basic text processing without spaCy")
                    self.nlp = None
                except Exception as e:
                    self.logger.error(f"Error downloading spaCy model: {e}")
                    self.logger.info("Falling back to basic text processing without spaCy")
                    self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Error setting up NLP components: {e}")
            raise
    
    def extract_username_from_url(self, profile_url: str) -> str:
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
    
    def fetch_user_data(self, username: str) -> Dict:
        """
        Fetch user posts and comments from Reddit with comprehensive data.
        
        Args:
            username: Reddit username
            
        Returns:
            Dictionary containing user data, posts, and comments with URLs
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
            for submission in redditor.submissions.new(limit=50):
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
            for comment in redditor.comments.new(limit=100):
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
    
    def clean_text(self, text: str) -> str:
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
    
    def extract_themes_with_spacy(self, texts: List[str]) -> Dict:
        """
        Extract themes, keywords, and characteristics using spaCy NLP.
        Falls back to basic text processing if spaCy is not available.
        
        Args:
            texts: List of cleaned text strings to analyze
            
        Returns:
            Dictionary containing extracted themes and characteristics
        """
        if not texts:
            return {}
        
        # Combine all texts for analysis
        combined_text = ' '.join(texts)
        
        # Initialize default values
        entities = {}
        key_phrases = []
        keywords = []
        
        if self.nlp is not None:
            # Process with spaCy if available
            try:
                doc = self.nlp(combined_text)
                
                # Extract named entities with categories
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    if ent.text not in entities[ent.label_]:
                        entities[ent.label_].append(ent.text)
                
                # Extract key phrases and noun chunks
                for chunk in doc.noun_chunks:
                    if 2 <= len(chunk.text.split()) <= 4:  # 2-4 word phrases
                        key_phrases.append(chunk.text.lower())
                
                # Extract keywords (important tokens)
                for token in doc:
                    if (not token.is_stop and not token.is_punct and 
                        token.is_alpha and len(token.text) > 3 and
                        token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                        keywords.append(token.lemma_.lower())
                        
            except Exception as e:
                self.logger.warning(f"Error processing with spaCy: {e}. Falling back to basic processing.")
                self.nlp = None
        
        # Basic text processing fallback if spaCy is not available
        if self.nlp is None:
            self.logger.info("Using basic text processing (spaCy not available)")
            
            # Simple keyword extraction using word frequency
            import string
            from collections import Counter
            
            # Remove punctuation and split into words
            translator = str.maketrans('', '', string.punctuation)
            words = combined_text.translate(translator).lower().split()
            
            # Filter out common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
            
            filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            keywords = [(word, count) for word, count in word_counts.most_common(15)]
            
            # Simple phrase extraction (bigrams and trigrams)
            phrases = []
            for i in range(len(filtered_words) - 1):
                bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                phrases.append(bigram)
            
            if len(filtered_words) > 2:
                for i in range(len(filtered_words) - 2):
                    trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                    phrases.append(trigram)
            
            phrase_counts = Counter(phrases)
            key_phrases = [phrase for phrase, count in phrase_counts.most_common(20) if count > 1]
        
        # Sentiment analysis using TextBlob
        sentiments = []
        for text in texts[:20]:  # Analyze first 20 texts for efficiency
            try:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            except Exception as e:
                self.logger.warning(f"Error in sentiment analysis: {e}")
                sentiments.append(0.0)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            'entities': entities,
            'key_phrases': key_phrases[:20],
            'keywords': keywords,
            'sentiment': avg_sentiment,
            'text_length': len(combined_text),
            'num_texts': len(texts)
        }
    
    def generate_persona_with_quantized_llm(self, user_data: Dict, themes: Dict) -> Dict:
        """
        Generate a comprehensive persona using the quantized LLM.
        
        Args:
            user_data: User data from Reddit
            themes: Extracted themes and characteristics
            
        Returns:
            Dictionary containing persona with citations
        """
        try:
            # Prepare context for the LLM
            context = self._prepare_llm_context(user_data, themes)
            
            # Create structured prompts for different persona aspects
            persona_aspects = {
                'demographics': self._generate_demographics_persona(context, user_data),
                'motivations': self._generate_motivations_persona(context),
                'personality': self._generate_personality_persona(context),
                'behavior': self._generate_behavior_persona(context),
                'frustrations': self._generate_frustrations_persona(context),
                'goals': self._generate_goals_persona(context),
                'quote': self._generate_quote_persona(context, user_data)
            }
            
            # Add citations to each aspect
            for aspect in persona_aspects:
                if isinstance(persona_aspects[aspect], dict) and 'description' in persona_aspects[aspect]:
                    persona_aspects[aspect]['citations'] = self._get_relevant_citations(
                        persona_aspects[aspect]['description'], user_data
                    )
            
            return persona_aspects
            
        except Exception as e:
            self.logger.error(f"Error generating persona: {e}")
            return {
                'demographics': {'description': f"Error generating demographics: {e}", 'citations': []},
                'motivations': {'description': f"Error generating motivations: {e}", 'citations': []},
                'personality': {'description': f"Error generating personality: {e}", 'citations': []},
                'behavior': {'description': f"Error generating behavior: {e}", 'citations': []},
                'frustrations': {'description': f"Error generating frustrations: {e}", 'citations': []},
                'goals': {'description': f"Error generating goals: {e}", 'citations': []},
                'quote': {'description': f"Error generating quote: {e}", 'citations': []}
            }
    
    def _prepare_llm_context(self, user_data: Dict, themes: Dict) -> str:
        """Prepare comprehensive context for the LLM with detailed user analysis."""
        context_parts = []
        
        # Basic user info
        context_parts.append(f"REDDIT USER ANALYSIS FOR: {user_data['username']}")
        context_parts.append(f"Account age: {self._get_account_age(user_data['created_utc'])}")
        context_parts.append(f"Activity level: {self._analyze_activity_level(user_data)}")
        context_parts.append(f"Engagement style: {self._analyze_engagement_style(user_data)}")
        
        # Detailed subreddit analysis
        subreddit_analysis = self._analyze_subreddit_participation(user_data)
        context_parts.append(f"Primary communities: {subreddit_analysis['primary_subs']}")
        context_parts.append(f"Community diversity: {subreddit_analysis['diversity_score']}")
        context_parts.append(f"Niche interests: {subreddit_analysis['niche_interests']}")
        
        # Content analysis
        content_analysis = self._analyze_content_patterns(user_data)
        context_parts.append(f"Writing style: {content_analysis['writing_style']}")
        context_parts.append(f"Common topics: {content_analysis['topics']}")
        context_parts.append(f"Language complexity: {content_analysis['complexity']}")
        
        # Behavioral patterns
        behavioral_analysis = self._analyze_behavioral_patterns(user_data)
        context_parts.append(f"Posting patterns: {behavioral_analysis['posting_patterns']}")
        context_parts.append(f"Interaction style: {behavioral_analysis['interaction_style']}")
        
        # Sentiment and emotional analysis
        context_parts.append(f"Overall sentiment: {themes.get('sentiment', 0):.2f}")
        context_parts.append(f"Emotional patterns: {self._analyze_emotional_patterns(user_data)}")
        
        # Most characteristic content
        context_parts.append("\nMOST CHARACTERISTIC CONTENT:")
        characteristic_content = self._get_most_characteristic_content(user_data)
        context_parts.extend(characteristic_content)
        
        return "\n".join(context_parts)
    
    def _get_account_age(self, created_utc: float) -> str:
        """Calculate and categorize account age."""
        from datetime import datetime
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
    
    def _analyze_activity_level(self, user_data: Dict) -> str:
        """Analyze user's activity level and patterns."""
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
    
    def _analyze_engagement_style(self, user_data: Dict) -> str:
        """Analyze how the user engages with Reddit."""
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
    
    def _analyze_subreddit_participation(self, user_data: Dict) -> Dict:
        """Analyze which subreddits the user participates in."""
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
        sorted_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize subreddits
        primary_subs = [sub for sub, count in sorted_subs[:5]]
        diversity_score = len(sorted_subs)
        
        # Identify niche interests
        niche_interests = self._identify_niche_interests(primary_subs)
        
        return {
            'primary_subs': ', '.join(primary_subs),
            'diversity_score': f"{diversity_score} different communities",
            'niche_interests': niche_interests
        }
    
    def _identify_niche_interests(self, subreddits: List[str]) -> str:
        """Identify niche interests based on subreddit names."""
        # Category mapping for common subreddits
        categories = {
            'tech': ['programming', 'technology', 'coding', 'python', 'javascript', 'webdev'],
            'gaming': ['gaming', 'games', 'minecraft', 'pokemon', 'rpg'],
            'fitness': ['fitness', 'bodybuilding', 'running', 'yoga'],
            'finance': ['investing', 'stocks', 'cryptocurrency', 'personalfinance'],
            'creative': ['art', 'music', 'photography', 'writing', 'design'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'space'],
            'lifestyle': ['fashion', 'food', 'cooking', 'travel', 'books'],
            'professional': ['career', 'entrepreneur', 'business', 'jobs']
        }
        
        identified_interests = []
        for category, keywords in categories.items():
            if any(keyword in sub.lower() for sub in subreddits for keyword in keywords):
                identified_interests.append(category)
        
        return ', '.join(identified_interests) if identified_interests else 'general interests'
    
    def _analyze_content_patterns(self, user_data: Dict) -> Dict:
        """Analyze patterns in user's content."""
        all_text = []
        
        # Collect all text content
        for post in user_data['posts']:
            if post['title']:
                all_text.append(post['title'])
            if post['body']:
                all_text.append(post['body'])
        
        for comment in user_data['comments']:
            if comment['body']:
                all_text.append(comment['body'])
        
        if not all_text:
            return {'writing_style': 'No content available', 'topics': 'None', 'complexity': 'Unknown'}
        
        # Analyze writing style
        writing_style = self._analyze_writing_style(all_text)
        
        # Extract topics
        topics = self._extract_topics(all_text)
        
        # Analyze complexity
        complexity = self._analyze_language_complexity(all_text)
        
        return {
            'writing_style': writing_style,
            'topics': topics,
            'complexity': complexity
        }
    
    def _analyze_writing_style(self, texts: List[str]) -> str:
        """Analyze writing style characteristics."""
        combined_text = ' '.join(texts)
        
        # Check for various style indicators
        avg_sentence_length = len(combined_text.split()) / max(len(combined_text.split('.')), 1)
        question_ratio = combined_text.count('?') / max(len(texts), 1)
        exclamation_ratio = combined_text.count('!') / max(len(texts), 1)
        caps_ratio = sum(1 for c in combined_text if c.isupper()) / max(len(combined_text), 1)
        
        style_traits = []
        
        if avg_sentence_length > 20:
            style_traits.append("verbose")
        elif avg_sentence_length < 8:
            style_traits.append("concise")
        
        if question_ratio > 0.3:
            style_traits.append("inquisitive")
        
        if exclamation_ratio > 0.2:
            style_traits.append("enthusiastic")
        
        if caps_ratio > 0.1:
            style_traits.append("emphatic")
        
        return ', '.join(style_traits) if style_traits else 'neutral'
    
    def _extract_topics(self, texts: List[str]) -> str:
        """Extract common topics from text."""
        combined_text = ' '.join(texts).lower()
        
        # Topic keywords
        topic_keywords = {
            'technology': ['tech', 'software', 'app', 'computer', 'internet', 'digital'],
            'work': ['job', 'work', 'career', 'office', 'boss', 'salary'],
            'relationships': ['relationship', 'dating', 'girlfriend', 'boyfriend', 'marriage'],
            'hobbies': ['hobby', 'collect', 'craft', 'build', 'create'],
            'health': ['health', 'fitness', 'exercise', 'diet', 'medical'],
            'entertainment': ['movie', 'show', 'game', 'music', 'book'],
            'politics': ['politics', 'government', 'election', 'policy'],
            'education': ['school', 'university', 'study', 'learn', 'education']
        }
        
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                found_topics.append(topic)
        
        return ', '.join(found_topics[:5]) if found_topics else 'general discussion'
    
    def _analyze_language_complexity(self, texts: List[str]) -> str:
        """Analyze language complexity."""
        combined_text = ' '.join(texts)
        words = combined_text.split()
        
        if not words:
            return 'Unknown'
        
        # Calculate average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
        
        # Check for complex indicators
        complex_indicators = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
        complex_count = sum(1 for word in words if word.lower() in complex_indicators)
        
        if avg_word_length > 6 or complex_count > 5:
            return 'Complex (sophisticated vocabulary)'
        elif avg_word_length > 4:
            return 'Moderate (clear communication)'
        else:
            return 'Simple (casual language)'
    
    def _analyze_behavioral_patterns(self, user_data: Dict) -> Dict:
        """Analyze behavioral patterns in posting."""
        # Analyze posting frequency
        posts_with_times = [(post['created_utc'], 'post') for post in user_data['posts']]
        comments_with_times = [(comment['created_utc'], 'comment') for comment in user_data['comments']]
        
        all_activity = sorted(posts_with_times + comments_with_times)
        
        # Analyze posting patterns
        posting_patterns = self._analyze_posting_times(all_activity)
        
        # Analyze interaction style
        interaction_style = self._analyze_interaction_style(user_data)
        
        return {
            'posting_patterns': posting_patterns,
            'interaction_style': interaction_style
        }
    
    def _analyze_posting_times(self, activity: List) -> str:
        """Analyze when user is most active."""
        if not activity:
            return 'No activity pattern'
        
        # Convert to hours
        hours = [datetime.fromtimestamp(timestamp).hour for timestamp, _ in activity]
        
        # Find peak hours
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 12
        
        if 6 <= peak_hour <= 11:
            return 'Morning poster (6-11 AM)'
        elif 12 <= peak_hour <= 17:
            return 'Afternoon poster (12-5 PM)'
        elif 18 <= peak_hour <= 22:
            return 'Evening poster (6-10 PM)'
        else:
            return 'Night owl (11 PM-5 AM)'
    
    def _analyze_interaction_style(self, user_data: Dict) -> str:
        """Analyze how user interacts with others."""
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
    
    def _analyze_emotional_patterns(self, user_data: Dict) -> str:
        """Analyze emotional patterns in content."""
        # Analyze sentiment per subreddit or topic
        emotions = []
        
        # Simple emotion detection
        positive_words = ['love', 'great', 'awesome', 'amazing', 'happy', 'excited']
        negative_words = ['hate', 'terrible', 'awful', 'frustrated', 'angry', 'sad']
        
        all_text = []
        for post in user_data['posts']:
            if post['title']:
                all_text.append(post['title'])
            if post['body']:
                all_text.append(post['body'])
        
        for comment in user_data['comments']:
            if comment['body']:
                all_text.append(comment['body'])
        
        if not all_text:
            return 'No emotional data'
        
        combined_text = ' '.join(all_text).lower()
        
        positive_count = sum(1 for word in positive_words if word in combined_text)
        negative_count = sum(1 for word in negative_words if word in combined_text)
        
        if positive_count > negative_count * 2:
            return 'Generally positive and optimistic'
        elif negative_count > positive_count * 2:
            return 'Often expresses frustration or criticism'
        else:
            return 'Balanced emotional expression'
    
    def _get_most_characteristic_content(self, user_data: Dict) -> List[str]:
        """Get the most characteristic content from the user."""
        characteristic_content = []
        
        # Get highest scored posts
        if user_data['posts']:
            top_posts = sorted(user_data['posts'], key=lambda x: x['score'], reverse=True)[:3]
            characteristic_content.append("Top posts:")
            for post in top_posts:
                characteristic_content.append(f"- '{post['title']}' (r/{post['subreddit']}, {post['score']} points)")
        
        # Get highest scored comments
        if user_data['comments']:
            top_comments = sorted(user_data['comments'], key=lambda x: x['score'], reverse=True)[:3]
            characteristic_content.append("Top comments:")
            for comment in top_comments:
                clean_comment = comment['body'][:100].replace('\n', ' ')
                characteristic_content.append(f"- '{clean_comment}...' (r/{comment['subreddit']}, {comment['score']} points)")
        
        return characteristic_content
    
    def _generate_demographics_persona(self, context: str, user_data: Dict) -> Dict:
        """Generate demographics persona characteristics."""
        prompt = f"""You are a skilled user researcher analyzing Reddit behavior patterns. Based on this comprehensive user analysis, estimate specific demographics:

{context}

CRITICAL ANALYSIS REQUIREMENTS:
1. Look for linguistic patterns, cultural references, and context clues
2. Analyze subreddit choices and posting patterns for occupation hints
3. Consider activity timing and engagement style for lifestyle insights
4. Use writing complexity and topics for education/age estimation
5. Examine interests and communities for geographic and social context

Based on the detailed analysis above, provide your best estimates:
- Age: Specific range based on language patterns, cultural references, and life stage indicators
- Occupation: Detailed category based on interests, communities, and knowledge displayed
- Status: Relationship hints from content, lifestyle, and posting patterns
- Location: Geographic hints from communities, timing, and cultural references
- Tier: Technology adoption level based on platform usage and interests
- Archetype: Primary personality type based on engagement and content patterns

Format as: Age: X, Occupation: Y, Status: Z, Location: W, Tier: V, Archetype: U"""
        
        response = self._query_llm(prompt, max_length=200)
        
        # Parse the response and create structured data
        demographics = {
            'age': self._infer_age_from_data(user_data),
            'occupation': self._infer_occupation_from_data(user_data),
            'status': self._infer_status_from_data(user_data),
            'location': self._infer_location_from_data(user_data),
            'tier': self._infer_tier_from_data(user_data),
            'archetype': self._infer_archetype_from_data(user_data)
        }
        
        # Try to extract structured data from response
        if response:
            lines = response.split(',')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in demographics:
                        demographics[key] = value
        
        return demographics
    
    def _infer_age_from_data(self, user_data: Dict) -> str:
        """Infer age range from user data patterns."""
        # Analyze account age and activity patterns
        account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
        account_years = account_age_days / 365
        
        # Analyze content for age indicators
        age_indicators = {
            'school': ['18-22', 'student'],
            'college': ['19-23', 'student'],
            'university': ['20-24', 'student'],
            'job': ['22-35', 'professional'],
            'career': ['25-40', 'professional'],
            'retirement': ['55-70', 'retiree'],
            'kids': ['28-45', 'parent'],
            'mortgage': ['30-50', 'homeowner']
        }
        
        # Check content for age clues
        all_text = []
        for post in user_data['posts']:
            if post['title']: all_text.append(post['title'])
            if post['body']: all_text.append(post['body'])
        for comment in user_data['comments']:
            if comment['body']: all_text.append(comment['body'])
        
        combined_text = ' '.join(all_text).lower()
        
        for indicator, (age_range, context) in age_indicators.items():
            if indicator in combined_text:
                return age_range
        
        # Default based on account age
        if account_years < 1:
            return "18-25"
        elif account_years < 3:
            return "22-30"
        elif account_years < 7:
            return "25-35"
        else:
            return "30-45"
    
    def _infer_occupation_from_data(self, user_data: Dict) -> str:
        """Infer occupation from subreddit participation and content."""
        # Analyze subreddits for occupation hints
        subreddits = []
        for post in user_data['posts']:
            subreddits.append(post['subreddit'].lower())
        for comment in user_data['comments']:
            subreddits.append(comment['subreddit'].lower())
        
        occupation_mapping = {
            'programming': 'Software Developer',
            'python': 'Software Developer',
            'javascript': 'Web Developer',
            'webdev': 'Web Developer',
            'cscareerquestions': 'Tech Professional',
            'entrepreneur': 'Business Owner',
            'investing': 'Finance Professional',
            'personalfinance': 'Finance Professional',
            'marketing': 'Marketing Professional',
            'design': 'Designer',
            'photography': 'Photographer/Creative',
            'writing': 'Writer',
            'teaching': 'Teacher',
            'medicine': 'Healthcare Professional',
            'law': 'Legal Professional',
            'engineering': 'Engineer',
            'science': 'Researcher/Scientist'
        }
        
        for sub in subreddits:
            for keyword, occupation in occupation_mapping.items():
                if keyword in sub:
                    return occupation
        
        return "Professional"
    
    def _infer_status_from_data(self, user_data: Dict) -> str:
        """Infer relationship status from content patterns."""
        all_text = []
        for post in user_data['posts']:
            if post['title']: all_text.append(post['title'])
            if post['body']: all_text.append(post['body'])
        for comment in user_data['comments']:
            if comment['body']: all_text.append(comment['body'])
        
        combined_text = ' '.join(all_text).lower()
        
        relationship_indicators = {
            'single': ['single', 'dating', 'tinder', 'bumble'],
            'relationship': ['girlfriend', 'boyfriend', 'partner', 'relationship'],
            'married': ['wife', 'husband', 'married', 'wedding'],
            'parent': ['kids', 'children', 'parenting', 'baby']
        }
        
        for status, keywords in relationship_indicators.items():
            if any(keyword in combined_text for keyword in keywords):
                return status.title()
        
        return "Single"
    
    def _infer_location_from_data(self, user_data: Dict) -> str:
        """Infer geographic location from content and communities."""
        # Look for location indicators in subreddits
        subreddits = []
        for post in user_data['posts']:
            subreddits.append(post['subreddit'].lower())
        for comment in user_data['comments']:
            subreddits.append(comment['subreddit'].lower())
        
        location_mapping = {
            'canada': 'Canada',
            'toronto': 'Canada',
            'vancouver': 'Canada',
            'unitedkingdom': 'UK',
            'london': 'UK',
            'australia': 'Australia',
            'sydney': 'Australia',
            'europe': 'Europe',
            'germany': 'Germany',
            'france': 'France',
            'nyc': 'US - Urban',
            'losangeles': 'US - Urban',
            'chicago': 'US - Urban',
            'sanfrancisco': 'US - Urban'
        }
        
        for sub in subreddits:
            for keyword, location in location_mapping.items():
                if keyword in sub:
                    return location
        
        return "Urban US"
    
    def _infer_tier_from_data(self, user_data: Dict) -> str:
        """Infer technology adoption tier from behavior."""
        # Analyze posting patterns and communities
        account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
        activity_level = len(user_data['posts']) + len(user_data['comments'])
        
        # Check for tech-savvy indicators
        tech_subs = ['programming', 'technology', 'gadgets', 'android', 'apple']
        subreddits = [post['subreddit'].lower() for post in user_data['posts']]
        subreddits.extend([comment['subreddit'].lower() for comment in user_data['comments']])
        
        tech_participation = sum(1 for sub in subreddits if any(tech in sub for tech in tech_subs))
        
        if tech_participation > 5 and account_age_days > 365:
            return "Early Adopter"
        elif tech_participation > 2:
            return "Mainstream"
        else:
            return "Conservative"
    
    def _infer_archetype_from_data(self, user_data: Dict) -> str:
        """Infer user archetype from engagement patterns."""
        posts = len(user_data['posts'])
        comments = len(user_data['comments'])
        
        if posts > comments:
            return "The Creator"
        elif comments > posts * 3:
            return "The Connector"
        else:
            return "The Explorer"
    
    def _generate_motivations_persona(self, context: str) -> Dict:
        """Generate motivations with levels."""
        prompt = f"""You are a behavioral psychologist analyzing Reddit user patterns. Based on this comprehensive user analysis, determine their core motivations:

{context}

DETAILED MOTIVATION ANALYSIS:
Analyze the user's behavior patterns, community choices, content themes, and engagement style to assess their motivations.

For each motivation, provide a rating (1-10) based on EVIDENCE from their behavior:

1. CONVENIENCE: Look for mentions of efficiency, time-saving, automation, frustration with complexity
2. WELLNESS: Check for health discussions, fitness communities, mental health awareness, self-care
3. SPEED: Analyze posting frequency, preference for quick interactions, impatience indicators
4. SOCIAL: Examine community engagement, relationship discussions, helping others, social issues
5. ACHIEVEMENT: Look for goal-setting, career discussions, skill development, accomplishment sharing
6. LEARNING: Check for educational content, how-to questions, knowledge sharing, curiosity

Be specific and evidence-based. Consider:
- Their primary subreddit choices reflect their values
- Writing style indicates personality priorities
- Activity patterns show what drives them
- Content themes reveal what matters most

Format as: Convenience: X, Wellness: Y, Speed: Z, Social: W, Achievement: V, Learning: U"""
        
        response = self._query_llm(prompt, max_length=250)
        
        # Default motivation levels with data-driven inference
        motivations = {
            'convenience': self._infer_convenience_motivation(context),
            'wellness': self._infer_wellness_motivation(context),
            'speed': self._infer_speed_motivation(context),
            'social': self._infer_social_motivation(context),
            'achievement': self._infer_achievement_motivation(context),
            'learning': self._infer_learning_motivation(context)
        }
        
        # Try to extract levels from response
        if response:
            lines = response.split(',')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    try:
                        level = int(value)
                        if key in motivations and 1 <= level <= 10:
                            motivations[key] = level
                    except ValueError:
                        pass
        
        return motivations
    
    def _infer_convenience_motivation(self, context: str) -> int:
        """Infer convenience motivation from context."""
        convenience_indicators = ['easy', 'simple', 'quick', 'efficient', 'automated', 'convenient']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in convenience_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _infer_wellness_motivation(self, context: str) -> int:
        """Infer wellness motivation from context."""
        wellness_indicators = ['health', 'fitness', 'wellness', 'mental health', 'exercise', 'diet']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in wellness_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _infer_speed_motivation(self, context: str) -> int:
        """Infer speed motivation from context."""
        speed_indicators = ['fast', 'quick', 'immediately', 'instant', 'rapid', 'speed']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in speed_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _infer_social_motivation(self, context: str) -> int:
        """Infer social motivation from context."""
        social_indicators = ['social', 'community', 'friends', 'relationship', 'connect', 'together']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in social_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _infer_achievement_motivation(self, context: str) -> int:
        """Infer achievement motivation from context."""
        achievement_indicators = ['goal', 'success', 'achieve', 'accomplish', 'career', 'progress']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in achievement_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _infer_learning_motivation(self, context: str) -> int:
        """Infer learning motivation from context."""
        learning_indicators = ['learn', 'study', 'education', 'knowledge', 'understand', 'research']
        context_lower = context.lower()
        score = 5  # base score
        
        for indicator in learning_indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _generate_personality_persona(self, context: str) -> Dict:
        """Generate personality traits with scales."""
        prompt = f"""You are a personality psychologist analyzing Reddit behavior patterns. Based on this comprehensive user analysis, determine their personality traits:

{context}

DETAILED PERSONALITY ANALYSIS:
Analyze the user's communication style, community engagement, content preferences, and behavioral patterns to assess their personality traits.

For each trait, provide a rating (1-10) based on BEHAVIORAL EVIDENCE:

1. EXTRAVERSION (1=Introverted, 10=Extraverted):
   - Look for social engagement, group participation, seeking attention
   - Introverts: lurk more, fewer but deeper comments, prefer one-on-one
   - Extraverts: active in large communities, frequent posting, seeks broad engagement

2. SENSING (1=Intuitive, 10=Sensing):
   - Look for practical vs theoretical discussions
   - Intuitives: abstract concepts, future-focused, big picture, possibilities
   - Sensors: concrete details, present-focused, step-by-step, facts

3. THINKING (1=Feeling, 10=Thinking):
   - Look for decision-making style and value priorities
   - Feeling: personal values, harmony, people-focused, subjective
   - Thinking: logical analysis, objective criteria, task-focused, impersonal

4. JUDGING (1=Perceiving, 10=Judging):
   - Look for planning vs spontaneity, structure vs flexibility
   - Perceiving: flexible, spontaneous, keep options open, adaptive
   - Judging: organized, planned, decisive, closure-seeking

Consider their posting patterns, language style, and community choices as key indicators.

Format as: Extraversion: X, Sensing: Y, Thinking: Z, Judging: W"""
        
        response = self._query_llm(prompt, max_length=250)
        
        # Default personality levels with data-driven inference
        personality = {
            'extraversion': self._infer_extraversion(context),
            'sensing': self._infer_sensing(context),
            'thinking': self._infer_thinking(context),
            'judging': self._infer_judging(context)
        }
        
        # Try to extract levels from response
        if response:
            lines = response.split(',')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    try:
                        level = int(value)
                        if key in personality and 1 <= level <= 10:
                            personality[key] = level
                    except ValueError:
                        pass
        
        return personality
    
    def _infer_extraversion(self, context: str) -> int:
        """Infer extraversion level from context."""
        context_lower = context.lower()
        score = 5  # base score
        
        # Look for social engagement indicators
        if 'very active' in context_lower or 'frequent' in context_lower:
            score += 2
        if 'heavy commenter' in context_lower or 'conversational' in context_lower:
            score += 2
        if 'community' in context_lower and 'diverse' in context_lower:
            score += 1
        if 'low activity' in context_lower or 'occasional' in context_lower:
            score -= 2
        
        return min(max(score, 1), 10)
    
    def _infer_sensing(self, context: str) -> int:
        """Infer sensing level from context."""
        context_lower = context.lower()
        score = 5  # base score
        
        # Look for practical vs theoretical indicators
        if 'practical' in context_lower or 'concrete' in context_lower:
            score += 2
        if 'how-to' in context_lower or 'step-by-step' in context_lower:
            score += 1
        if 'abstract' in context_lower or 'theoretical' in context_lower:
            score -= 2
        if 'future' in context_lower or 'possibilities' in context_lower:
            score -= 1
        
        return min(max(score, 1), 10)
    
    def _infer_thinking(self, context: str) -> int:
        """Infer thinking level from context."""
        context_lower = context.lower()
        score = 5  # base score
        
        # Look for logical vs emotional indicators
        if 'tech' in context_lower or 'programming' in context_lower:
            score += 2
        if 'logical' in context_lower or 'analysis' in context_lower:
            score += 1
        if 'relationship' in context_lower or 'feelings' in context_lower:
            score -= 2
        if 'emotional' in context_lower or 'personal' in context_lower:
            score -= 1
        
        return min(max(score, 1), 10)
    
    def _infer_judging(self, context: str) -> int:
        """Infer judging level from context."""
        context_lower = context.lower()
        score = 5  # base score
        
        # Look for structure vs flexibility indicators
        if 'organized' in context_lower or 'planned' in context_lower:
            score += 2
        if 'morning poster' in context_lower or 'regular' in context_lower:
            score += 1
        if 'flexible' in context_lower or 'spontaneous' in context_lower:
            score -= 2
        if 'night owl' in context_lower or 'random' in context_lower:
            score -= 1
        
        return min(max(score, 1), 10)
    
    def _generate_frustrations_persona(self, context: str) -> Dict:
        """Generate frustrations as bullet points."""
        prompt = f"""You are analyzing a Reddit user's actual frustrations from their behavior patterns. Based on this user analysis, identify their REAL frustrations:

{context}

CRITICAL INSTRUCTIONS:
- Only mention frustrations that are EVIDENT from their actual Reddit activity
- Focus on specific problems they've actually experienced or complained about
- Avoid generic suggestions or template language
- Be concise and specific to this user's situation
- No instructional text or generic advice

Generate 3-5 specific frustrations this user actually faces:"""
        
        response = self._query_llm(prompt, max_length=300)
        
        # Split response into bullet points
        frustrations = []
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    frustrations.append(line.lstrip('•-* '))
                elif line and len(frustrations) < 5:
                    frustrations.append(line)
        
        # Add context-based frustrations if response is limited
        if len(frustrations) < 3:
            context_frustrations = self._infer_frustrations_from_context(context)
            frustrations.extend(context_frustrations)
        
        return {'items': frustrations[:5]}
    
    def _infer_frustrations_from_context(self, context: str) -> List[str]:
        """Infer frustrations from context patterns."""
        frustrations = []
        context_lower = context.lower()
        
        # Technology frustrations
        if 'tech' in context_lower or 'programming' in context_lower:
            frustrations.append("Debugging code issues and dealing with system crashes")
        
        # Social frustrations
        if 'low activity' in context_lower:
            frustrations.append("Shallow online interactions and lack of genuine discussion")
        
        # Efficiency frustrations
        if 'convenience' in context_lower:
            frustrations.append("Repetitive tasks that could be automated")
        
        # Information frustrations
        if 'learning' in context_lower:
            frustrations.append("Sifting through outdated tutorials and conflicting advice")
        
        return frustrations[:3]
    
    def _generate_quote_persona(self, context: str, user_data: Dict = None) -> str:
        """Generate a representative quote."""
        # Analyze actual user language patterns if available
        language_analysis = ""
        if user_data:
            language_analysis = self._analyze_user_language_patterns(user_data)
        
        # Enhanced prompt for more authentic quotes
        prompt = f"""Create an authentic quote that captures this Reddit user's voice and mindset:

{context}

{language_analysis}

QUOTE REQUIREMENTS:
- Sound natural and conversational, like something they'd actually say
- Reflect their specific interests, personality, and communication style
- Be 8-15 words maximum for impact
- Use their actual language patterns and tone
- Avoid generic advice or inspirational clichés
- Make it personal and specific to their situation

Examples of authentic quotes:
- "I spend way too much time debugging code, but I love the challenge"
- "Finding genuine conversations online is harder than it should be"
- "I'm always looking for ways to optimize my workflow"

Generate ONE authentic quote (no quotes around it):"""
        
        response = self._query_llm_for_quote(prompt, max_length=100)
        
        # Clean and extract quote
        if response:
            # Remove quotes if present
            cleaned_response = response.strip().strip('"').strip("'")
            
            # Remove common artifacts
            cleaned_response = self._clean_quote_response(cleaned_response)
            
            # Validate quote quality
            if self._is_valid_quote(cleaned_response):
                return cleaned_response
        
        # Generate context-based quote if LLM fails
        return self._generate_enhanced_context_quote(context)
    
    def _clean_quote_response(self, response: str) -> str:
        """Clean quote response by removing artifacts and improving quality."""
        # Remove common quote artifacts
        quote_artifacts = [
            'generate one authentic quote',
            'based on the analysis',
            'this user would say',
            'authentic quote that',
            'captures their',
            'represents their',
            'reflects their'
        ]
        
        # Split into sentences and find the best one
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Filter out sentences with artifacts
        clean_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            is_artifact = any(artifact in sentence_lower for artifact in quote_artifacts)
            if not is_artifact and len(sentence) > 5:
                clean_sentences.append(sentence)
        
        # Return the first clean sentence or the original if none found
        if clean_sentences:
            return clean_sentences[0].strip()
        return response.strip()
    
    def _is_valid_quote(self, quote: str) -> bool:
        """Validate if a quote is high quality and authentic."""
        quote_lower = quote.lower()
        
        # Check length
        if len(quote) < 8 or len(quote) > 100:
            return False
        
        # Check for artifacts
        invalid_patterns = [
            'generate',
            'create',
            'authentic',
            'quote',
            'based on',
            'analysis',
            'user would',
            'this captures',
            'represents',
            'reflects'
        ]
        
        if any(pattern in quote_lower for pattern in invalid_patterns):
            return False
        
        # Check for first person language (more authentic)
        first_person_indicators = ['i ', 'my ', 'me ', 'myself', "i'm", "i've", "i'd", "i'll"]
        has_first_person = any(indicator in quote_lower for indicator in first_person_indicators)
        
        # Check for conversational language
        conversational_indicators = ['really', 'actually', 'honestly', 'basically', 'pretty', 'kinda', 'sorta']
        has_conversational = any(indicator in quote_lower for indicator in conversational_indicators)
        
        # Prefer quotes with first person or conversational language
        return has_first_person or has_conversational or len(quote.split()) <= 12
    
    def _generate_enhanced_context_quote(self, context: str) -> str:
        """Generate enhanced context-based quotes with better personalization."""
        context_lower = context.lower()
        
        # Analyze user characteristics for more personalized quotes
        user_traits = self._analyze_user_traits_for_quote(context_lower)
        
        # Generate quotes based on multiple factors
        if 'tech' in context_lower and 'programming' in context_lower:
            if 'frustrated' in context_lower:
                return "Debugging is 90% of programming, and I'm getting really good at it"
            elif 'learning' in context_lower:
                return "Every bug I fix teaches me something new about the system"
            else:
                return "Clean code is satisfying, but working code pays the bills"
        
        elif 'gaming' in context_lower:
            if 'competitive' in context_lower:
                return "I play to win, but I'm here for the challenge more than anything"
            else:
                return "Games are my escape from the real world's problems"
        
        elif 'social' in context_lower or 'community' in context_lower:
            if 'frustrated' in context_lower:
                return "Real conversations are rare online, but worth searching for"
            else:
                return "The best communities are built on shared interests, not just likes"
        
        elif 'learning' in context_lower or 'education' in context_lower:
            return "I learn best by doing, not just reading about it"
        
        elif 'creative' in context_lower or 'art' in context_lower:
            return "Creating something from nothing is the best feeling in the world"
        
        elif 'fitness' in context_lower or 'health' in context_lower:
            return "Small consistent efforts beat big dramatic changes every time"
        
        elif 'business' in context_lower or 'entrepreneur' in context_lower:
            return "Ideas are worthless without execution, but execution needs direction"
        
        elif 'finance' in context_lower or 'investing' in context_lower:
            return "Time in the market beats timing the market, usually"
        
        # Activity-based quotes
        elif 'very active' in context_lower:
            return "I'm always curious about what people are thinking and sharing"
        
        elif 'low activity' in context_lower:
            return "I prefer to listen more than I speak, online and off"
        
        elif 'night owl' in context_lower:
            return "My best thinking happens when the world is quiet"
        
        elif 'morning poster' in context_lower:
            return "Starting the day with fresh perspectives from the community"
        
        # Personality-based quotes
        elif user_traits.get('optimistic', False):
            return "There's always a solution, sometimes you just need a different angle"
        
        elif user_traits.get('analytical', False):
            return "I like to understand the why behind the what"
        
        elif user_traits.get('helpful', False):
            return "If I can save someone else time, that's a win for everyone"
        
        elif user_traits.get('critical', False):
            return "Most things could be done better, and I'm not afraid to say it"
        
        # Default contextual quotes
        else:
            return "Quality over quantity in everything I do"
    
    def _analyze_user_traits_for_quote(self, context: str) -> dict:
        """Analyze user traits from context to generate better quotes."""
        traits = {}
        
        # Optimism indicators
        optimism_words = ['great', 'awesome', 'love', 'amazing', 'excited', 'positive']
        traits['optimistic'] = any(word in context for word in optimism_words)
        
        # Analytical indicators
        analytical_words = ['analysis', 'data', 'research', 'study', 'examine', 'logical']
        traits['analytical'] = any(word in context for word in analytical_words)
        
        # Helpful indicators
        helpful_words = ['help', 'assist', 'support', 'advice', 'guide', 'explain']
        traits['helpful'] = any(word in context for word in helpful_words)
        
        # Critical indicators
        critical_words = ['problem', 'issue', 'wrong', 'bad', 'terrible', 'frustrated']
        traits['critical'] = any(word in context for word in critical_words)
        
        return traits
    
    def _query_llm_for_quote(self, prompt: str, max_length: int = 100) -> str:
        """
        Query the quantized LLM specifically for quote generation with optimized parameters.
        
        Args:
            prompt: Input prompt for the LLM
            max_length: Maximum response length
            
        Returns:
            Generated quote text
        """
        try:
            # Tokenize input with better truncation handling
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            
            # Move inputs to the same device as the model
            if hasattr(self, 'device') and self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with parameters optimized for quotes
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.9,  # Higher temperature for more creativity
                    do_sample=True,
                    top_p=0.85,  # Nucleus sampling for natural language
                    top_k=40,  # Slightly lower top-k for more focused results
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.2,  # Higher repetition penalty for quotes
                    length_penalty=0.8,  # Favor shorter, punchier quotes
                    no_repeat_ngram_size=3  # Prevent repetitive phrases
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            response = response[len(prompt):].strip()
            
            # Clean up response specifically for quotes
            response = self._clean_quote_llm_response(response)
            
            return response if response else "Every day is a chance to learn something new."
            
        except Exception as e:
            self.logger.error(f"Error querying LLM for quote: {e}")
            return "Focus on progress, not perfection."
    
    def _clean_quote_llm_response(self, response: str) -> str:
        """Clean up LLM response specifically for quotes."""
        # Remove common artifacts
        response = response.replace('<|endoftext|>', '')
        response = response.replace('</s>', '')
        response = response.strip()
        
        # Remove quotes if they wrap the entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        
        # Take only the first sentence for quotes
        sentences = response.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if first_sentence and len(first_sentence) > 8:
                return first_sentence
        
        # Take only the first line if multiple lines
        lines = response.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) > 8:
                return first_line
        
        return response.strip()
    
    def _analyze_user_language_patterns(self, user_data: Dict) -> str:
        """Analyze user's actual language patterns for better quote generation."""
        if not user_data:
            return ""
        
        # Collect actual user text
        user_texts = []
        
        # Add post titles and content
        for post in user_data.get('posts', [])[:10]:  # Analyze recent posts
            if post.get('title'):
                user_texts.append(post['title'])
            if post.get('body'):
                user_texts.append(post['body'])
        
        # Add comments
        for comment in user_data.get('comments', [])[:20]:  # Analyze recent comments
            if comment.get('body'):
                user_texts.append(comment['body'])
        
        if not user_texts:
            return ""
        
        # Analyze language patterns
        analysis_parts = []
        
        # Common phrases they use
        common_phrases = self._extract_common_phrases(user_texts)
        if common_phrases:
            analysis_parts.append(f"Common phrases they use: {', '.join(common_phrases[:5])}")
        
        # Conversation starters they use
        conversation_starters = self._extract_conversation_starters(user_texts)
        if conversation_starters:
            analysis_parts.append(f"How they start conversations: {', '.join(conversation_starters[:3])}")
        
        # Their typical language style
        language_style = self._analyze_language_style(user_texts)
        if language_style:
            analysis_parts.append(f"Language style: {language_style}")
        
        # Emotional expressions they use
        emotional_expressions = self._extract_emotional_expressions(user_texts)
        if emotional_expressions:
            analysis_parts.append(f"Emotional expressions: {', '.join(emotional_expressions[:3])}")
        
        if analysis_parts:
            return "USER'S ACTUAL LANGUAGE PATTERNS:\n" + "\n".join(analysis_parts)
        
        return ""
    
    def _extract_common_phrases(self, texts: List[str]) -> List[str]:
        """Extract commonly used phrases from user texts."""
        # Combine all texts
        combined = ' '.join(texts).lower()
        
        # Look for common expressions
        common_expressions = [
            "i think", "i feel", "i believe", "in my opinion", "personally", "honestly",
            "to be honest", "actually", "really", "pretty much", "kind of", "sort of",
            "you know", "i mean", "basically", "literally", "definitely", "probably",
            "obviously", "clearly", "certainly", "absolutely", "totally", "completely"
        ]
        
        found_phrases = []
        for phrase in common_expressions:
            if phrase in combined:
                found_phrases.append(phrase)
        
        return found_phrases
    
    def _extract_conversation_starters(self, texts: List[str]) -> List[str]:
        """Extract how the user typically starts conversations."""
        starters = []
        
        for text in texts:
            text_lower = text.lower().strip()
            
            # Look for question starters
            if text_lower.startswith(('what', 'how', 'why', 'when', 'where', 'which', 'who')):
                # Extract the first few words
                words = text_lower.split()[:4]
                if len(words) >= 2:
                    starters.append(' '.join(words))
            
            # Look for statement starters
            elif text_lower.startswith(('i', 'my', 'this', 'that', 'there', 'here')):
                words = text_lower.split()[:3]
                if len(words) >= 2:
                    starters.append(' '.join(words))
        
        # Return most common starters
        from collections import Counter
        if starters:
            counter = Counter(starters)
            return [starter for starter, count in counter.most_common(5) if count > 1]
        
        return []
    
    def _analyze_language_style(self, texts: List[str]) -> str:
        """Analyze the user's overall language style."""
        combined = ' '.join(texts).lower()
        
        style_indicators = []
        
        # Check for casual language
        casual_words = ['lol', 'lmao', 'haha', 'yeah', 'yep', 'nah', 'gonna', 'wanna', 'kinda']
        if any(word in combined for word in casual_words):
            style_indicators.append("casual")
        
        # Check for technical language
        tech_words = ['algorithm', 'function', 'variable', 'database', 'server', 'code', 'programming']
        if any(word in combined for word in tech_words):
            style_indicators.append("technical")
        
        # Check for analytical language
        analytical_words = ['analysis', 'data', 'research', 'study', 'evidence', 'statistics']
        if any(word in combined for word in analytical_words):
            style_indicators.append("analytical")
        
        # Check for enthusiastic language
        enthusiastic_words = ['awesome', 'amazing', 'fantastic', 'incredible', 'love', 'excited']
        if any(word in combined for word in enthusiastic_words):
            style_indicators.append("enthusiastic")
        
        # Check for formal language
        formal_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently']
        if any(word in combined for word in formal_words):
            style_indicators.append("formal")
        
        return ', '.join(style_indicators) if style_indicators else "conversational"
    
    def _extract_emotional_expressions(self, texts: List[str]) -> List[str]:
        """Extract emotional expressions the user commonly uses."""
        combined = ' '.join(texts).lower()
        
        # Common emotional expressions
        emotional_patterns = [
            "i'm excited", "i'm frustrated", "i'm confused", "i'm happy", "i'm sad",
            "love this", "hate this", "this is great", "this sucks", "not sure",
            "i feel like", "makes me", "i'm tired", "i'm done", "can't believe"
        ]
        
        found_expressions = []
        for pattern in emotional_patterns:
            if pattern in combined:
                found_expressions.append(pattern)
        
        return found_expressions
    
    def _generate_context_based_quote(self, context: str) -> str:
        """Generate a quote based on context patterns (legacy method)."""
        # Use the enhanced context quote method
        return self._generate_enhanced_context_quote(context)
    
    def _generate_behavior_persona(self, context: str) -> Dict:
        """Generate behavior-related persona characteristics."""
        prompt = f"""You are analyzing a Reddit user's actual behavioral patterns. Based on this user analysis, describe their SPECIFIC behaviors:

{context}

CRITICAL INSTRUCTIONS:
- Only mention behaviors that are EVIDENT from their actual Reddit activity
- Focus on specific patterns in their posting and interaction style
- Avoid generic suggestions or template language
- Be concise and specific to this user's situation
- No instructional text or generic advice

Generate 3-5 specific behavioral patterns this user actually exhibits:"""
        
        response = self._query_llm(prompt, max_length=300)
        
        # Split response into bullet points
        behaviors = []
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    behaviors.append(line.lstrip('•-* '))
                elif line and len(behaviors) < 5:
                    behaviors.append(line)
        
        # Add context-based behaviors if response is limited
        if len(behaviors) < 3:
            context_behaviors = self._infer_behaviors_from_context(context)
            behaviors.extend(context_behaviors)
        
        return {'items': behaviors[:5]}
    
    def _infer_behaviors_from_context(self, context: str) -> List[str]:
        """Infer behaviors from context patterns."""
        behaviors = []
        context_lower = context.lower()
        
        # Activity patterns
        if 'very active' in context_lower:
            behaviors.append("Checks Reddit multiple times daily and engages with current discussions")
        elif 'low activity' in context_lower:
            behaviors.append("Browses casually and only comments when truly interested")
        
        # Engagement patterns
        if 'heavy commenter' in context_lower:
            behaviors.append("Prefers detailed discussions over quick reactions")
        elif 'content creator' in context_lower:
            behaviors.append("Shares original content and initiates conversations")
        
        # Communication style
        if 'enthusiastic' in context_lower:
            behaviors.append("Writes with energy and uses emphatic language")
        elif 'concise' in context_lower:
            behaviors.append("Gets straight to the point in responses")
        
        # Time patterns
        if 'morning poster' in context_lower:
            behaviors.append("Most active during morning routines and commute")
        elif 'night owl' in context_lower:
            behaviors.append("Engages more during evening relaxation time")
        
        return behaviors[:3]
    
    def _generate_goals_persona(self, context: str) -> Dict:
        """Generate goals-related persona characteristics."""
        prompt = f"""You are analyzing a Reddit user's actual goals from their behavior patterns. Based on this user analysis, identify their REAL goals and needs:

{context}

CRITICAL INSTRUCTIONS:
- Only mention goals that are EVIDENT from their actual Reddit activity
- Focus on what they're actively pursuing or asking about
- Avoid generic suggestions or template language
- Be concise and specific to this user's situation
- No instructional text or generic advice

Generate 3-5 specific goals this user is actually working toward:"""
        
        response = self._query_llm(prompt, max_length=300)
        
        # Split response into bullet points
        goals = []
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    goals.append(line.lstrip('•-* '))
                elif line and len(goals) < 5:
                    goals.append(line)
        
        # Add context-based goals if response is limited
        if len(goals) < 3:
            context_goals = self._infer_goals_from_context(context)
            goals.extend(context_goals)
        
        return {'items': goals[:5]}
    
    def _infer_goals_from_context(self, context: str) -> List[str]:
        """Infer goals from context patterns."""
        goals = []
        context_lower = context.lower()
        
        # Career/professional goals
        if 'tech' in context_lower or 'programming' in context_lower:
            goals.append("Master new frameworks and build more efficient solutions")
        
        # Learning goals
        if 'learning' in context_lower or 'education' in context_lower:
            goals.append("Develop expertise in areas that genuinely interest them")
        
        # Social goals
        if 'social' in context_lower or 'community' in context_lower:
            goals.append("Find like-minded people to collaborate and share ideas with")
        
        # Health/wellness goals
        if 'wellness' in context_lower or 'health' in context_lower:
            goals.append("Create sustainable habits for long-term health")
        
        # Efficiency goals
        if 'convenience' in context_lower or 'speed' in context_lower:
            goals.append("Streamline processes to focus on what matters most")
        
        return goals[:3]
    
    def _query_llm(self, prompt: str, max_length: int = 200) -> str:
        """
        Query the quantized LLM with a prompt.
        Optimized for GPU usage with enhanced generation parameters.
        
        Args:
            prompt: Input prompt for the LLM
            max_length: Maximum response length
            
        Returns:
            Generated response text
        """
        try:
            # Tokenize input with better truncation handling
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            
            # Move inputs to the same device as the model
            if hasattr(self, 'device') and self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with enhanced parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.8,  # Slightly higher for more creativity
                    do_sample=True,
                    top_p=0.9,  # Nucleus sampling for better quality
                    top_k=50,  # Top-k sampling
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for better performance
                    repetition_penalty=1.1,  # Reduce repetition
                    length_penalty=1.0  # Neutral length penalty
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            response = response[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_llm_response(response)
            
            return response if response else "Unable to generate response."
            
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            return f"Error generating response: {e}"
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing artifacts and generic content."""
        # Remove common artifacts
        response = response.replace('<|endoftext|>', '')
        response = response.replace('</s>', '')
        response = response.strip()
        
        # Remove placeholder text and generic instructions
        artifact_patterns = [
            'this content is being provided for informational purposes',
            'please check with your local community',
            'set goals by category',
            'add objectives by year',
            'include other categories if you have them',
            'create a user profile that has',
            'some things to consider',
            'quote text here',
            'format as:',
            'format as',
            'based on the analysis above',
            'detailed analysis',
            'evidence from their behavior',
            'word cards',
            'posters that highlight',
            'customize to give a clear picture',
            'identify 3-5 specific',
            'generate 3-5 specific',
            'critical instructions',
            'consider the following'
        ]
        
        # Filter out lines containing artifacts
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that contain artifact patterns
            is_artifact = any(pattern in line_lower for pattern in artifact_patterns)
            if is_artifact:
                continue
                
            # Skip repetitive lines
            if line not in cleaned_lines:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _get_relevant_citations(self, description: str, user_data: Dict) -> List[str]:
        """
        Find relevant citations (URLs) for a persona description.
        
        Args:
            description: Generated persona description
            user_data: User data containing posts and comments
            
        Returns:
            List of relevant URLs
        """
        citations = []
        
        # Simple keyword matching for citations
        description_lower = description.lower()
        
        # Check posts for relevance
        for post in user_data['posts'][:10]:  # Check first 10 posts
            post_text = f"{post['title']} {post.get('body', '')}".lower()
            if any(word in post_text for word in description_lower.split() if len(word) > 3):
                citations.append(post['url'])
        
        # Check comments for relevance
        for comment in user_data['comments'][:10]:  # Check first 10 comments
            comment_text = comment['body'].lower()
            if any(word in comment_text for word in description_lower.split() if len(word) > 3):
                citations.append(comment['url'])
        
        return citations[:3]  # Return top 3 citations
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format Unix timestamp to readable date."""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    
    def _format_keywords(self, keywords: List) -> str:
        """Format keywords list, handling both tuple and string formats."""
        if not keywords:
            return "None"
        
        if keywords and isinstance(keywords[0], tuple):
            return ', '.join([word for word, _ in keywords])
        else:
            return ', '.join([str(word) for word in keywords])
    
    def generate_persona_report(self, profile_url: str, output_file: str = None) -> Dict:
        """
        Generate a complete persona report for a Reddit user.
        
        Args:
            profile_url: Reddit profile URL
            output_file: Optional output file path
            
        Returns:
            Dictionary containing persona data and file path
        """
        try:
            # Extract username
            username = self.extract_username_from_url(profile_url)
            self.logger.info(f"Processing user: {username}")
            
            # Fetch user data
            user_data = self.fetch_user_data(username)
            
            # Check if user has any content
            if not user_data['posts'] and not user_data['comments']:
                return {
                    'error': f"User '{username}' has no public posts or comments available for analysis.",
                    'username': username,
                    'output_file': None
                }
            
            # Clean and prepare text data
            all_texts = []
            for post in user_data['posts']:
                if post['title']:
                    all_texts.append(self.clean_text(post['title']))
                if post['body']:
                    all_texts.append(self.clean_text(post['body']))
            
            for comment in user_data['comments']:
                if comment['body']:
                    all_texts.append(self.clean_text(comment['body']))
            
            # Extract themes using spaCy
            self.logger.info("Extracting themes and characteristics...")
            themes = self.extract_themes_with_spacy(all_texts)
            
            # Generate persona using quantized LLM
            self.logger.info("Generating persona with quantized LLM...")
            persona = self.generate_persona_with_quantized_llm(user_data, themes)
            
            # Create output file
            if not output_file:
                output_file = f"output/persona_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Write formatted report
            report_content = self._format_persona_report(username, persona, user_data, themes)
            
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
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating persona report: {e}"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'username': 'unknown',
                'output_file': None
            }
    
    def _format_persona_report(self, username: str, persona: Dict, 
                             user_data: Dict, themes: Dict) -> str:
        """Format the persona report in the requested format."""
        report_parts = [
            f"User Persona for {username}:",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Profile URL: https://www.reddit.com/user/{username}/",
            "",
            "=" * 60,
            ""
        ]
        
        # Add demographics
        if 'demographics' in persona:
            demo = persona['demographics']
            report_parts.extend([
                "DEMOGRAPHICS:",
                f"- Age: {demo.get('age', 'Unknown')}",
                f"- Occupation: {demo.get('occupation', 'Unknown')}",
                f"- Status: {demo.get('status', 'Unknown')}",
                f"- Location: {demo.get('location', 'Unknown')}",
                f"- Tier: {demo.get('tier', 'Unknown')}",
                f"- Archetype: {demo.get('archetype', 'Unknown')}",
                ""
            ])
        
        # Add motivations
        if 'motivations' in persona:
            report_parts.append("MOTIVATIONS:")
            for motivation, level in persona['motivations'].items():
                report_parts.append(f"- {motivation.title()}: {level}/10")
            report_parts.append("")
        
        # Add personality
        if 'personality' in persona:
            report_parts.append("PERSONALITY:")
            for trait, level in persona['personality'].items():
                report_parts.append(f"- {trait.title()}: {level}/10")
            report_parts.append("")
        
        # Add behavior
        if 'behavior' in persona:
            report_parts.append("BEHAVIOUR & HABITS:")
            if 'items' in persona['behavior']:
                for item in persona['behavior']['items']:
                    report_parts.append(f"- {item}")
            else:
                report_parts.append("- No specific behaviors identified")
            report_parts.append("")
        
        # Add frustrations
        if 'frustrations' in persona:
            report_parts.append("FRUSTRATIONS:")
            if 'items' in persona['frustrations']:
                for item in persona['frustrations']['items']:
                    report_parts.append(f"- {item}")
            else:
                report_parts.append("- No specific frustrations identified")
            report_parts.append("")
        
        # Add goals
        if 'goals' in persona:
            report_parts.append("GOALS & NEEDS:")
            if 'items' in persona['goals']:
                for item in persona['goals']['items']:
                    report_parts.append(f"- {item}")
            else:
                report_parts.append("- No specific goals identified")
            report_parts.append("")
        
        # Add quote
        if 'quote' in persona:
            report_parts.extend([
                "REPRESENTATIVE QUOTE:",
                f'"{persona["quote"]}"',
                ""
            ])
        
        # Add statistics
        report_parts.extend([
            "=" * 60,
            "USER STATISTICS:",
            f"- Account Created: {self._format_timestamp(user_data['created_utc'])}",
            f"- Comment Karma: {user_data['comment_karma']:,}",
            f"- Link Karma: {user_data['link_karma']:,}",
            f"- Total Posts: {len(user_data['posts'])}",
            f"- Total Comments: {len(user_data['comments'])}",
            "",
            "NLP ANALYSIS:",
            f"- Overall Sentiment: {themes.get('sentiment', 0):.3f} (-1 to 1 scale)",
            f"- Text Analyzed: {themes.get('text_length', 0):,} characters",
            f"- Top Keywords: {self._format_keywords(themes.get('keywords', [])[:10])}",
            ""
        ])
        
        return "\n".join(report_parts)


# Flask Application
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Global generator instance
generator = None

def initialize_generator():
    """Initialize the persona generator with environment variables."""
    global generator
    
    load_dotenv()
    
    reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
    reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'PersonaGenerator/1.0')
    
    if not all([reddit_client_id, reddit_client_secret]):
        raise ValueError("Missing required Reddit API credentials")
    
    generator = QuantizedLLMPersonaGenerator(
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent
    )

@app.route('/')
def index():
    """Serve a simple web interface for testing."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reddit Persona Generator</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f8f9fa; 
                color: #333;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                margin-bottom: 30px;
            }
            .form-section {
                margin-bottom: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            input[type="text"] { 
                width: 100%; 
                padding: 12px; 
                margin: 10px 0; 
                border: 2px solid #e9ecef;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button { 
                background: #e67e22; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
            }
            button:hover { background: #d35400; }
            .persona-card {
                display: none;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin-top: 30px;
                overflow: hidden;
            }
            .persona-header {
                background: linear-gradient(135deg, #e67e22, #f39c12);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .persona-name {
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .persona-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                padding: 30px;
            }
            .persona-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
            }
            .section-title {
                font-size: 1.4em;
                font-weight: 600;
                color: #e67e22;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .demographics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 20px;
            }
            .demo-item {
                background: white;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #e67e22;
            }
            .demo-label {
                font-weight: 600;
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .demo-value {
                font-size: 1.1em;
                color: #2c3e50;
                margin-top: 4px;
            }
            .motivation-bar {
                display: flex;
                align-items: center;
                margin-bottom: 12px;
            }
            .motivation-label {
                width: 120px;
                font-weight: 600;
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
            }
            .motivation-progress {
                flex: 1;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                margin: 0 15px;
                overflow: hidden;
            }
            .motivation-fill {
                height: 100%;
                background: linear-gradient(90deg, #e67e22, #f39c12);
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            .motivation-value {
                font-weight: 600;
                color: #2c3e50;
                min-width: 25px;
            }
            .personality-trait {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            .trait-left {
                width: 90px;
                text-align: right;
                font-size: 0.9em;
                color: #666;
                font-weight: 600;
            }
            .trait-slider {
                flex: 1;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                margin: 0 15px;
                position: relative;
                overflow: hidden;
            }
            .trait-indicator {
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                width: 16px;
                height: 16px;
                background: #e67e22;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            .trait-right {
                width: 90px;
                font-size: 0.9em;
                color: #666;
                font-weight: 600;
            }
            .bullet-list {
                list-style: none;
                padding: 0;
            }
            .bullet-list li {
                background: white;
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 6px;
                border-left: 4px solid #e67e22;
                position: relative;
            }
            .bullet-list li::before {
                content: "•";
                color: #e67e22;
                font-weight: bold;
                position: absolute;
                left: -8px;
            }
            .quote-section {
                grid-column: 1 / -1;
                background: #2c3e50;
                color: white;
                padding: 30px;
                border-radius: 8px;
                text-align: center;
                font-style: italic;
                font-size: 1.3em;
                position: relative;
            }
            .quote-section::before {
                content: '"';
                font-size: 4em;
                color: #e67e22;
                position: absolute;
                top: -10px;
                left: 20px;
            }
            .quote-section::after {
                content: '"';
                font-size: 4em;
                color: #e67e22;
                position: absolute;
                bottom: -40px;
                right: 20px;
            }
            .error { 
                background: #ffe6e6; 
                color: #cc0000; 
                padding: 20px; 
                border-radius: 8px; 
                margin-top: 20px;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .avatar-placeholder {
                width: 120px;
                height: 120px;
                background: #bdc3c7;
                border-radius: 50%;
                margin: 0 auto 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 3em;
                color: white;
            }
            @media (max-width: 768px) {
                .persona-content {
                    grid-template-columns: 1fr;
                }
                .demographics-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reddit Persona Generator</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Enter a Reddit user profile URL to generate an AI-powered persona analysis.
            </p>
            
            <div class="form-section">
                <form id="personaForm">
                    <input type="text" id="profileUrl" placeholder="https://www.reddit.com/user/username/" required>
                    <button type="submit">Generate Persona</button>
                </form>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('personaForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const url = document.getElementById('profileUrl').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<div class="loading">Generating persona... This may take a few minutes.</div>';
                
                try {
                    const response = await fetch('/generate_persona', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ profile_url: url })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${data.error}</div>`;
                    } else {
                        displayPersonaCard(data);
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
                }
            });
            
            function displayPersonaCard(data) {
                const resultDiv = document.getElementById('result');
                const persona = data.persona;
                
                const personaHtml = `
                    <div class="persona-card" style="display: block;">
                        <div class="persona-header">
                            <div class="avatar-placeholder">
                                ${data.username.charAt(0).toUpperCase()}
                            </div>
                            <div class="persona-name">${data.username}</div>
                        </div>
                        
                        <div class="persona-content">
                            <div class="persona-section">
                                <div class="section-title">Demographics</div>
                                <div class="demographics-grid">
                                    <div class="demo-item">
                                        <div class="demo-label">Age</div>
                                        <div class="demo-value">${persona.demographics.age}</div>
                                    </div>
                                    <div class="demo-item">
                                        <div class="demo-label">Occupation</div>
                                        <div class="demo-value">${persona.demographics.occupation}</div>
                                    </div>
                                    <div class="demo-item">
                                        <div class="demo-label">Status</div>
                                        <div class="demo-value">${persona.demographics.status}</div>
                                    </div>
                                    <div class="demo-item">
                                        <div class="demo-label">Location</div>
                                        <div class="demo-value">${persona.demographics.location}</div>
                                    </div>
                                    <div class="demo-item">
                                        <div class="demo-label">Tier</div>
                                        <div class="demo-value">${persona.demographics.tier}</div>
                                    </div>
                                    <div class="demo-item">
                                        <div class="demo-label">Archetype</div>
                                        <div class="demo-value">${persona.demographics.archetype}</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="persona-section">
                                <div class="section-title">Motivations</div>
                                ${Object.entries(persona.motivations).map(([key, value]) => `
                                    <div class="motivation-bar">
                                        <div class="motivation-label">${key}</div>
                                        <div class="motivation-progress">
                                            <div class="motivation-fill" style="width: ${value * 10}%"></div>
                                        </div>
                                        <div class="motivation-value">${value}</div>
                                    </div>
                                `).join('')}
                            </div>
                            
                            <div class="persona-section">
                                <div class="section-title">Personality</div>
                                ${createPersonalityTraits(persona.personality)}
                            </div>
                            
                                                         <div class="persona-section">
                                 <div class="section-title">Behaviour & Habits</div>
                                 <ul class="bullet-list">
                                     ${persona.behavior.items ? persona.behavior.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific behaviors identified</li>'}
                                 </ul>
                             </div>
                            
                            <div class="persona-section">
                                <div class="section-title">Frustrations</div>
                                <ul class="bullet-list">
                                    ${persona.frustrations.items ? persona.frustrations.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific frustrations identified</li>'}
                                </ul>
                            </div>
                            
                                                         <div class="persona-section">
                                 <div class="section-title">Goals & Needs</div>
                                 <ul class="bullet-list">
                                     ${persona.goals.items ? persona.goals.items.map(item => `<li>${item}</li>`).join('') : '<li>No specific goals identified</li>'}
                                 </ul>
                             </div>
                            
                            <div class="quote-section">
                                ${persona.quote}
                            </div>
                        </div>
                    </div>
                `;
                
                resultDiv.innerHTML = personaHtml;
            }
            
            function createPersonalityTraits(personality) {
                const traits = [
                    { left: 'Introvert', right: 'Extrovert', key: 'extraversion' },
                    { left: 'Intuition', right: 'Sensing', key: 'sensing' },
                    { left: 'Feeling', right: 'Thinking', key: 'thinking' },
                    { left: 'Perceiving', right: 'Judging', key: 'judging' }
                ];
                
                return traits.map(trait => `
                    <div class="personality-trait">
                        <div class="trait-left">${trait.left}</div>
                        <div class="trait-slider">
                            <div class="trait-indicator" style="left: ${((personality[trait.key] || 5) - 1) * 10}%"></div>
                        </div>
                        <div class="trait-right">${trait.right}</div>
                    </div>
                `).join('');
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/generate_persona', methods=['POST'])
def generate_persona_api():
    """
    Flask API endpoint to generate persona from Reddit profile URL.
    
    Expected JSON payload:
    {
        "profile_url": "https://www.reddit.com/user/username/"
    }
    
    Returns:
    JSON response with persona data or error message
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'profile_url' not in data:
            return jsonify({
                'error': 'Missing profile_url in request body'
            }), 400
        
        profile_url = data['profile_url']
        
        # Validate URL format
        if not profile_url or 'reddit.com/user/' not in profile_url:
            return jsonify({
                'error': 'Invalid Reddit profile URL format'
            }), 400
        
        # Generate persona
        result = generator.generate_persona_report(profile_url)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': generator is not None
    })

def main():
    """
    Main function to run the Reddit Persona Generator.
    
    This function can be used to:
    1. Run the Flask API server
    2. Process individual URLs directly
    3. Run test cases
    """
    print("Reddit Persona Generator with Quantized LLM")
    print("=" * 50)
    
    # Initialize the generator
    try:
        initialize_generator()
        print("✓ Generator initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing generator: {e}")
        print("\nTroubleshooting:")
        if "protobuf" in str(e).lower() or "messagefactory" in str(e).lower():
            print("🚨 PROTOBUF COMPATIBILITY ISSUE DETECTED!")
            print("Quick fix: Run 'python fix_protobuf.py'")
            print("Or manually: pip install protobuf==3.20.3 transformers==4.24.0")
        else:
            print("Please ensure you have:")
            print("1. Valid Reddit API credentials in .env file")
            print("2. All required dependencies installed")
            print("3. Sufficient GPU memory for quantized LLM")
        print("\nFor detailed help, see TROUBLESHOOTING.md")
        return
    
    # Test with sample URLs
    test_urls = [
        "https://www.reddit.com/user/kojied/",
        "https://www.reddit.com/user/Hungry-Move-6603/"
    ]
    
    print("\nTesting with sample URLs:")
    print("-" * 30)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\nProcessing URL {i}: {url}")
        
        try:
            result = generator.generate_persona_report(url)
            
            if 'error' in result:
                print(f"✗ Error: {result['error']}")
            else:
                print(f"✓ Success: {result['username']}")
                print(f"  Output file: {result['output_file']}")
                print(f"  Posts: {result['user_stats']['posts']}")
                print(f"  Comments: {result['user_stats']['comments']}")
                
        except Exception as e:
            print(f"✗ Error processing {url}: {e}")
    
    print("\n" + "=" * 50)
    print("Starting Flask API server...")
    print("Access the web interface at: http://localhost:5000")
    print("API endpoint: POST /generate_persona")
    print("Press Ctrl+C to stop the server")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main() 
