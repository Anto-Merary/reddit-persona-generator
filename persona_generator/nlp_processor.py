"""
NLP Processing module for the Persona Generator.

This module handles text analysis, sentiment analysis, and natural language processing
using spaCy and other NLP libraries.
"""

import logging
import subprocess
from typing import Dict, List, Optional
from collections import Counter
import string
import spacy
import nltk
from textblob import TextBlob
from config import Config
from .utils import clean_text


class NLPProcessor:
    """Natural Language Processing handler for text analysis."""
    
    def __init__(self):
        """Initialize the NLP processor."""
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self._setup_nlp_components()
    
    def _setup_nlp_components(self):
        """Initialize NLP components including spaCy and NLTK."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load spaCy model with better error handling
            nlp_config = Config.get_nlp_config()
            spacy_model = nlp_config['spacy_model']
            
            try:
                self.nlp = spacy.load(spacy_model)
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                self.logger.warning(f"spaCy model '{spacy_model}' not found. Attempting to download...")
                self._download_spacy_model(spacy_model)
                
        except Exception as e:
            self.logger.error(f"Error setting up NLP components: {e}")
            raise
    
    def _download_spacy_model(self, model_name: str):
        """Download spaCy model if not available."""
        try:
            # Try to download the model
            result = subprocess.run(
                ["python", "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("spaCy model downloaded successfully")
            self.nlp = spacy.load(model_name)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download spaCy model: {e}")
            self.logger.info("Falling back to basic text processing without spaCy")
            self.nlp = None
        except Exception as e:
            self.logger.error(f"Error downloading spaCy model: {e}")
            self.logger.info("Falling back to basic text processing without spaCy")
            self.nlp = None
    
    def extract_themes_and_entities(self, texts: List[str]) -> Dict:
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
        
        # Clean all texts
        cleaned_texts = [clean_text(text) for text in texts if text]
        
        # Combine all texts for analysis
        combined_text = ' '.join(cleaned_texts)
        
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
            translator = str.maketrans('', '', string.punctuation)
            words = combined_text.translate(translator).lower().split()
            
            # Filter out common stop words and short words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
            
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
        for text in cleaned_texts[:20]:  # Analyze first 20 texts for efficiency
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
            'num_texts': len(cleaned_texts)
        }
    
    def analyze_content_patterns(self, user_data: Dict) -> Dict:
        """
        Analyze patterns in user's content.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Dictionary with content analysis results
        """
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
            return {
                'writing_style': 'No content available',
                'topics': 'None',
                'complexity': 'Unknown',
                'emotional_patterns': 'No data'
            }
        
        # Analyze writing style
        writing_style = self._analyze_writing_style(all_text)
        
        # Extract topics
        topics = self._extract_topics(all_text)
        
        # Analyze complexity
        complexity = self._analyze_language_complexity(all_text)
        
        # Analyze emotional patterns
        emotional_patterns = self._analyze_emotional_patterns(all_text)
        
        return {
            'writing_style': writing_style,
            'topics': topics,
            'complexity': complexity,
            'emotional_patterns': emotional_patterns
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
    
    def _analyze_emotional_patterns(self, texts: List[str]) -> str:
        """Analyze emotional patterns in content."""
        # Simple emotion detection
        positive_words = ['love', 'great', 'awesome', 'amazing', 'happy', 'excited']
        negative_words = ['hate', 'terrible', 'awful', 'frustrated', 'angry', 'sad']
        
        if not texts:
            return 'No emotional data'
        
        combined_text = ' '.join(texts).lower()
        
        positive_count = sum(1 for word in positive_words if word in combined_text)
        negative_count = sum(1 for word in negative_words if word in combined_text)
        
        if positive_count > negative_count * 2:
            return 'Generally positive and optimistic'
        elif negative_count > positive_count * 2:
            return 'Often expresses frustration or criticism'
        else:
            return 'Balanced emotional expression'
    
    def extract_user_language_patterns(self, user_data: Dict) -> Dict:
        """
        Analyze user's actual language patterns for better persona generation.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Dictionary with language pattern analysis
        """
        if not user_data:
            return {}
        
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
            return {}
        
        # Analyze language patterns
        analysis = {
            'common_phrases': self._extract_common_phrases(user_texts),
            'conversation_starters': self._extract_conversation_starters(user_texts),
            'language_style': self._analyze_language_style(user_texts),
            'emotional_expressions': self._extract_emotional_expressions(user_texts)
        }
        
        return analysis
    
    def _extract_common_phrases(self, texts: List[str]) -> List[str]:
        """Extract commonly used phrases from user texts."""
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
                words = text_lower.split()[:4]
                if len(words) >= 2:
                    starters.append(' '.join(words))
            
            # Look for statement starters
            elif text_lower.startswith(('i', 'my', 'this', 'that', 'there', 'here')):
                words = text_lower.split()[:3]
                if len(words) >= 2:
                    starters.append(' '.join(words))
        
        # Return most common starters
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