"""
Persona Builder module for the Persona Generator.

This module handles the construction of comprehensive persona profiles
from analyzed user data, NLP insights, and LLM-generated content.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter
from .utils import (
    get_account_age, analyze_activity_level, analyze_engagement_style,
    get_relevant_citations, format_timestamp, format_keywords
)


class PersonaBuilder:
    """Builder class for constructing comprehensive user personas."""
    
    def __init__(self, llm_handler, nlp_processor, reddit_client):
        """
        Initialize the persona builder.
        
        Args:
            llm_handler: LLM handler instance
            nlp_processor: NLP processor instance
            reddit_client: Reddit client instance
        """
        self.logger = logging.getLogger(__name__)
        self.llm_handler = llm_handler
        self.nlp_processor = nlp_processor
        self.reddit_client = reddit_client
    
    def build_persona(self, user_data: Dict, themes: Dict) -> Dict:
        """
        Build a comprehensive persona from user data and themes.
        
        Args:
            user_data: User data from Reddit
            themes: NLP-extracted themes and characteristics
            
        Returns:
            Dictionary containing complete persona
        """
        try:
            self.logger.info("Building comprehensive persona...")
            
            # Prepare context for persona generation
            context = self._prepare_context(user_data, themes)
            
            # Generate different aspects of the persona
            persona_aspects = {
                'demographics': self._build_demographics(context, user_data),
                'motivations': self._build_motivations(context),
                'personality': self._build_personality(context),
                'behavior': self._build_behavior(context),
                'frustrations': self._build_frustrations(context),
                'goals': self._build_goals(context),
                'quote': self._build_quote(context, user_data)
            }
            
            # Add citations to each aspect
            self._add_citations(persona_aspects, user_data)
            
            self.logger.info("Persona built successfully")
            return persona_aspects
            
        except Exception as e:
            self.logger.error(f"Error building persona: {e}")
            return self._create_error_persona(str(e))
    
    def _prepare_context(self, user_data: Dict, themes: Dict) -> str:
        """Prepare comprehensive context for persona generation."""
        context_parts = []
        
        # Basic user info
        context_parts.append(f"REDDIT USER ANALYSIS FOR: {user_data['username']}")
        context_parts.append(f"Account age: {get_account_age(user_data['created_utc'])}")
        context_parts.append(f"Activity level: {analyze_activity_level(user_data)}")
        context_parts.append(f"Engagement style: {analyze_engagement_style(user_data)}")
        
        # Subreddit analysis
        subreddit_analysis = self._analyze_subreddit_participation(user_data)
        context_parts.append(f"Primary communities: {subreddit_analysis['primary_subs']}")
        context_parts.append(f"Community diversity: {subreddit_analysis['diversity_score']}")
        context_parts.append(f"Niche interests: {subreddit_analysis['niche_interests']}")
        
        # Content analysis
        content_analysis = self.nlp_processor.analyze_content_patterns(user_data)
        context_parts.append(f"Writing style: {content_analysis['writing_style']}")
        context_parts.append(f"Common topics: {content_analysis['topics']}")
        context_parts.append(f"Language complexity: {content_analysis['complexity']}")
        
        # Behavioral patterns
        behavioral_analysis = self.reddit_client.analyze_posting_patterns(user_data)
        context_parts.append(f"Posting patterns: {behavioral_analysis['activity_pattern']}")
        context_parts.append(f"Interaction style: {self.reddit_client.get_interaction_style(user_data)}")
        
        # Sentiment and emotional analysis
        context_parts.append(f"Overall sentiment: {themes.get('sentiment', 0):.2f}")
        context_parts.append(f"Emotional patterns: {content_analysis['emotional_patterns']}")
        
        # Most characteristic content
        context_parts.append("\nMOST CHARACTERISTIC CONTENT:")
        characteristic_content = self._get_most_characteristic_content(user_data)
        context_parts.extend(characteristic_content)
        
        return "\n".join(context_parts)
    
    def _analyze_subreddit_participation(self, user_data: Dict) -> Dict:
        """Analyze which subreddits the user participates in."""
        subreddit_counts = self.reddit_client.get_user_subreddits(user_data)
        
        # Get top subreddits
        primary_subs = list(subreddit_counts.keys())[:5]
        diversity_score = len(subreddit_counts)
        
        # Identify niche interests
        niche_interests = self._identify_niche_interests(primary_subs)
        
        return {
            'primary_subs': ', '.join(primary_subs),
            'diversity_score': f"{diversity_score} different communities",
            'niche_interests': niche_interests
        }
    
    def _identify_niche_interests(self, subreddits: List[str]) -> str:
        """Identify niche interests based on subreddit names."""
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
    
    def _get_most_characteristic_content(self, user_data: Dict) -> List[str]:
        """Get the most characteristic content from the user."""
        characteristic_content = []
        
        # Get top content
        top_content = self.reddit_client.get_top_content(user_data, limit=3)
        
        if top_content:
            characteristic_content.append("Top content:")
            for item in top_content:
                if item['type'] == 'post':
                    characteristic_content.append(f"- '{item['title']}' (r/{item['subreddit']}, {item['score']} points)")
                else:
                    clean_comment = item['body'][:100].replace('\n', ' ')
                    characteristic_content.append(f"- '{clean_comment}...' (r/{item['subreddit']}, {item['score']} points)")
        
        return characteristic_content
    
    def _build_demographics(self, context: str, user_data: Dict) -> Dict:
        """Build demographics aspect of persona."""
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
        
        response = self.llm_handler.generate_text(prompt, max_length=200)
        
        # Parse response and create structured demographics
        demographics = self._parse_demographics_response(response, user_data)
        
        return demographics
    
    def _parse_demographics_response(self, response: str, user_data: Dict) -> Dict:
        """Parse demographics response from LLM."""
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
        account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
        account_years = account_age_days / 365
        
        # Check content for age clues
        age_indicators = {
            'school': '18-22',
            'college': '19-23',
            'university': '20-24',
            'job': '22-35',
            'career': '25-40',
            'retirement': '55-70',
            'kids': '28-45',
            'mortgage': '30-50'
        }
        
        # Analyze text for age indicators
        all_text = []
        for post in user_data['posts']:
            if post['title']: all_text.append(post['title'])
            if post['body']: all_text.append(post['body'])
        for comment in user_data['comments']:
            if comment['body']: all_text.append(comment['body'])
        
        combined_text = ' '.join(all_text).lower()
        
        for indicator, age_range in age_indicators.items():
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
        """Infer occupation from subreddit participation."""
        subreddits = self.reddit_client.get_user_subreddits(user_data)
        
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
        
        for sub in subreddits.keys():
            for keyword, occupation in occupation_mapping.items():
                if keyword in sub.lower():
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
            'Single': ['single', 'dating', 'tinder', 'bumble'],
            'Relationship': ['girlfriend', 'boyfriend', 'partner', 'relationship'],
            'Married': ['wife', 'husband', 'married', 'wedding'],
            'Parent': ['kids', 'children', 'parenting', 'baby']
        }
        
        for status, keywords in relationship_indicators.items():
            if any(keyword in combined_text for keyword in keywords):
                return status
        
        return "Single"
    
    def _infer_location_from_data(self, user_data: Dict) -> str:
        """Infer geographic location from communities."""
        subreddits = self.reddit_client.get_user_subreddits(user_data)
        
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
        
        for sub in subreddits.keys():
            for keyword, location in location_mapping.items():
                if keyword in sub.lower():
                    return location
        
        return "Urban US"
    
    def _infer_tier_from_data(self, user_data: Dict) -> str:
        """Infer technology adoption tier from behavior."""
        account_age_days = (datetime.now().timestamp() - user_data['created_utc']) / 86400
        activity_level = len(user_data['posts']) + len(user_data['comments'])
        
        # Check for tech-savvy indicators
        tech_subs = ['programming', 'technology', 'gadgets', 'android', 'apple']
        subreddits = self.reddit_client.get_user_subreddits(user_data)
        
        tech_participation = sum(1 for sub in subreddits.keys() if any(tech in sub.lower() for tech in tech_subs))
        
        if tech_participation > 2 and account_age_days > 365:
            return "Early Adopter"
        elif tech_participation > 0:
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
    
    def _build_motivations(self, context: str) -> Dict:
        """Build motivations aspect of persona."""
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

Format as: Convenience: X, Wellness: Y, Speed: Z, Social: W, Achievement: V, Learning: U"""
        
        response = self.llm_handler.generate_text(prompt, max_length=250)
        
        # Parse response and create structured motivations
        motivations = self._parse_motivations_response(response, context)
        
        return motivations
    
    def _parse_motivations_response(self, response: str, context: str) -> Dict:
        """Parse motivations response from LLM."""
        # Default motivation levels with inference
        motivations = {
            'convenience': self._infer_motivation_level(context, 'convenience'),
            'wellness': self._infer_motivation_level(context, 'wellness'),
            'speed': self._infer_motivation_level(context, 'speed'),
            'social': self._infer_motivation_level(context, 'social'),
            'achievement': self._infer_motivation_level(context, 'achievement'),
            'learning': self._infer_motivation_level(context, 'learning')
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
    
    def _infer_motivation_level(self, context: str, motivation_type: str) -> int:
        """Infer motivation level from context."""
        context_lower = context.lower()
        
        motivation_indicators = {
            'convenience': ['easy', 'simple', 'quick', 'efficient', 'automated'],
            'wellness': ['health', 'fitness', 'wellness', 'mental health', 'exercise'],
            'speed': ['fast', 'quick', 'immediately', 'instant', 'rapid'],
            'social': ['social', 'community', 'friends', 'relationship', 'connect'],
            'achievement': ['goal', 'success', 'achieve', 'accomplish', 'career'],
            'learning': ['learn', 'study', 'education', 'knowledge', 'understand']
        }
        
        indicators = motivation_indicators.get(motivation_type, [])
        score = 5  # base score
        
        for indicator in indicators:
            if indicator in context_lower:
                score += 1
        
        return min(score, 10)
    
    def _build_personality(self, context: str) -> Dict:
        """Build personality aspect of persona."""
        prompt = f"""You are a personality psychologist analyzing Reddit behavior patterns. Based on this comprehensive user analysis, determine their personality traits:

{context}

DETAILED PERSONALITY ANALYSIS:
Analyze the user's communication style, community engagement, content preferences, and behavioral patterns to assess their personality traits.

For each trait, provide a rating (1-10) based on BEHAVIORAL EVIDENCE:

1. EXTRAVERSION (1=Introverted, 10=Extraverted): Social engagement, group participation
2. SENSING (1=Intuitive, 10=Sensing): Practical vs theoretical discussions
3. THINKING (1=Feeling, 10=Thinking): Decision-making style and value priorities
4. JUDGING (1=Perceiving, 10=Judging): Planning vs spontaneity, structure vs flexibility

Format as: Extraversion: X, Sensing: Y, Thinking: Z, Judging: W"""
        
        response = self.llm_handler.generate_text(prompt, max_length=250)
        
        # Parse response and create structured personality
        personality = self._parse_personality_response(response, context)
        
        return personality
    
    def _parse_personality_response(self, response: str, context: str) -> Dict:
        """Parse personality response from LLM."""
        # Default personality levels with inference
        personality = {
            'extraversion': self._infer_personality_trait(context, 'extraversion'),
            'sensing': self._infer_personality_trait(context, 'sensing'),
            'thinking': self._infer_personality_trait(context, 'thinking'),
            'judging': self._infer_personality_trait(context, 'judging')
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
    
    def _infer_personality_trait(self, context: str, trait: str) -> int:
        """Infer personality trait level from context."""
        context_lower = context.lower()
        score = 5  # base score
        
        if trait == 'extraversion':
            if 'very active' in context_lower or 'heavy commenter' in context_lower:
                score += 2
            if 'low activity' in context_lower:
                score -= 2
        elif trait == 'sensing':
            if 'practical' in context_lower or 'concrete' in context_lower:
                score += 2
            if 'abstract' in context_lower or 'theoretical' in context_lower:
                score -= 2
        elif trait == 'thinking':
            if 'tech' in context_lower or 'logical' in context_lower:
                score += 2
            if 'emotional' in context_lower or 'personal' in context_lower:
                score -= 2
        elif trait == 'judging':
            if 'organized' in context_lower or 'planned' in context_lower:
                score += 2
            if 'flexible' in context_lower or 'spontaneous' in context_lower:
                score -= 2
        
        return min(max(score, 1), 10)
    
    def _build_behavior(self, context: str) -> Dict:
        """Build behavior aspect of persona."""
        prompt = f"""Based on this Reddit user's actual behavior patterns, describe their specific behaviors. Be concise and avoid repetition.

{context}

List exactly 3-4 distinct behavioral patterns this user exhibits based on their Reddit activity:
1."""
        
        response = self.llm_handler.generate_text(prompt, max_length=250)
        
        # Parse response into behavior items
        behaviors = self._parse_numbered_response(response)
        
        # Add context-based behaviors if needed
        if len(behaviors) < 3:
            context_behaviors = self._infer_behaviors_from_context(context)
            behaviors.extend(context_behaviors)
        
        return {'items': behaviors[:5]}
    
    def _build_frustrations(self, context: str) -> Dict:
        """Build frustrations aspect of persona."""
        prompt = f"""Based on this Reddit user's actual behavior patterns, identify their specific frustrations. Be concise and avoid repetition.

{context}

List exactly 3-4 distinct frustrations this user faces based on their Reddit activity:
1."""
        
        response = self.llm_handler.generate_text(prompt, max_length=250)
        
        # Parse response into frustration items
        frustrations = self._parse_numbered_response(response)
        
        # Add context-based frustrations if needed
        if len(frustrations) < 3:
            context_frustrations = self._infer_frustrations_from_context(context)
            frustrations.extend(context_frustrations)
        
        return {'items': frustrations[:5]}
    
    def _build_goals(self, context: str) -> Dict:
        """Build goals aspect of persona."""
        prompt = f"""Based on this Reddit user's actual behavior patterns, identify their specific goals and needs. Be concise and avoid repetition.

{context}

List exactly 3-4 distinct goals this user is working toward based on their Reddit activity:
1."""
        
        response = self.llm_handler.generate_text(prompt, max_length=250)
        
        # Parse response into goal items
        goals = self._parse_numbered_response(response)
        
        # Add context-based goals if needed
        if len(goals) < 3:
            context_goals = self._infer_goals_from_context(context)
            goals.extend(context_goals)
        
        return {'items': goals[:5]}
    
    def _build_quote(self, context: str, user_data: Dict) -> str:
        """Build representative quote for persona."""
        # Analyze user's language patterns
        language_patterns = self.nlp_processor.extract_user_language_patterns(user_data)
        
        # Create enhanced prompt with language analysis
        language_analysis = ""
        if language_patterns:
            language_analysis = f"USER'S LANGUAGE PATTERNS:\n"
            for key, value in language_patterns.items():
                if value:
                    language_analysis += f"{key}: {value}\n"
        
        prompt = f"""Create an authentic quote that captures this Reddit user's voice and mindset:

{context}

{language_analysis}

Generate ONE authentic quote (8-15 words, natural and conversational):"""
        
        response = self.llm_handler.generate_text(prompt, max_length=100, for_quote=True)
        
        # Validate and clean quote
        if response and self._is_valid_quote(response):
            return response
        
        # Generate context-based quote if LLM fails
        return self._generate_context_quote(context)
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse list response from LLM."""
        items = []
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    items.append(line.lstrip('•-* '))
                elif line and len(items) < 5:
                    items.append(line)
        return items
    
    def _parse_numbered_response(self, response: str) -> List[str]:
        """Parse numbered response from LLM with enhanced artifact filtering."""
        items = []
        if not response:
            return items
        
        # First, add the initial "1." if it exists
        if response.strip():
            response = "1. " + response.strip()
        
        # Split into lines and process
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for numbered items (1., 2., 3., etc.)
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Extract content after the number
                content = line[2:].strip()
                if content and self._is_valid_content(content):
                    items.append(content)
            # Check for bullet points as fallback
            elif line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                content = line.lstrip('•-* ').strip()
                if content and self._is_valid_content(content):
                    items.append(content)
            # Check for plain text items if we don't have many items yet
            elif line and len(items) < 3 and self._is_valid_content(line):
                items.append(line)
        
        return items[:5]
    
    def _is_valid_content(self, content: str) -> bool:
        """Check if content is valid and not an artifact."""
        if not content or len(content) < 10:
            return False
        
        # Check for common artifacts
        artifacts = [
            'generate', 'create', 'based on', 'analysis', 'user would',
            'this captures', 'represents', 'reflects', 'critical instructions',
            'format as', 'consider the following', 'detailed analysis',
            'evidence from', 'specific frustrations', 'behavioral patterns',
            'working toward', 'actually faces', 'exhibits based'
        ]
        
        content_lower = content.lower()
        if any(artifact in content_lower for artifact in artifacts):
            return False
        
        # Check for excessive repetition
        words = content.split()
        if len(words) > 3:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 2 times in a short sentence, it's likely repetitive
            if any(count > 2 for count in word_counts.values()):
                return False
        
        return True
    
    def _infer_behaviors_from_context(self, context: str) -> List[str]:
        """Infer behaviors from context patterns."""
        behaviors = []
        context_lower = context.lower()
        
        if 'very active' in context_lower:
            behaviors.append("Checks Reddit multiple times daily and engages with discussions")
        elif 'low activity' in context_lower:
            behaviors.append("Browses casually and only comments when truly interested")
        
        if 'heavy commenter' in context_lower:
            behaviors.append("Prefers detailed discussions over quick reactions")
        elif 'content creator' in context_lower:
            behaviors.append("Shares original content and initiates conversations")
        
        return behaviors[:3]
    
    def _infer_frustrations_from_context(self, context: str) -> List[str]:
        """Infer frustrations from context patterns."""
        frustrations = []
        context_lower = context.lower()
        
        if 'tech' in context_lower:
            frustrations.append("Debugging code issues and dealing with system crashes")
        
        if 'low activity' in context_lower:
            frustrations.append("Shallow online interactions and lack of genuine discussion")
        
        if 'convenience' in context_lower:
            frustrations.append("Repetitive tasks that could be automated")
        
        return frustrations[:3]
    
    def _infer_goals_from_context(self, context: str) -> List[str]:
        """Infer goals from context patterns."""
        goals = []
        context_lower = context.lower()
        
        if 'tech' in context_lower:
            goals.append("Master new frameworks and build more efficient solutions")
        
        if 'learning' in context_lower:
            goals.append("Develop expertise in areas that genuinely interest them")
        
        if 'social' in context_lower:
            goals.append("Find like-minded people to collaborate and share ideas with")
        
        return goals[:3]
    
    def _is_valid_quote(self, quote: str) -> bool:
        """Validate if a quote is high quality."""
        quote_lower = quote.lower()
        
        # Check length
        if len(quote) < 8 or len(quote) > 100:
            return False
        
        # Check for artifacts
        invalid_patterns = ['generate', 'create', 'authentic', 'quote', 'based on', 'analysis']
        if any(pattern in quote_lower for pattern in invalid_patterns):
            return False
        
        return True
    
    def _generate_context_quote(self, context: str) -> str:
        """Generate context-based quote as fallback."""
        context_lower = context.lower()
        
        if 'tech' in context_lower and 'programming' in context_lower:
            return "Clean code is satisfying, but working code pays the bills"
        elif 'gaming' in context_lower:
            return "Games are my escape from the real world's problems"
        elif 'social' in context_lower:
            return "Real conversations are rare online, but worth searching for"
        elif 'learning' in context_lower:
            return "I learn best by doing, not just reading about it"
        else:
            return "Quality over quantity in everything I do"
    
    def _add_citations(self, persona_aspects: Dict, user_data: Dict):
        """Add citations to persona aspects."""
        for aspect_name, aspect_data in persona_aspects.items():
            if isinstance(aspect_data, dict) and 'items' in aspect_data:
                # For list-based aspects
                aspect_data['citations'] = get_relevant_citations(
                    ' '.join(aspect_data['items']), user_data
                )
            elif isinstance(aspect_data, dict):
                # For dictionary-based aspects
                description = str(aspect_data)
                aspect_data['citations'] = get_relevant_citations(description, user_data)
            elif isinstance(aspect_data, str):
                # For string-based aspects (quotes)
                citations = get_relevant_citations(aspect_data, user_data)
                # Convert to dict format for consistency
                persona_aspects[aspect_name] = {
                    'description': aspect_data,
                    'citations': citations
                }
    
    def _create_error_persona(self, error_message: str) -> Dict:
        """Create error persona when generation fails."""
        return {
            'demographics': {'error': f"Error generating demographics: {error_message}"},
            'motivations': {'error': f"Error generating motivations: {error_message}"},
            'personality': {'error': f"Error generating personality: {error_message}"},
            'behavior': {'error': f"Error generating behavior: {error_message}"},
            'frustrations': {'error': f"Error generating frustrations: {error_message}"},
            'goals': {'error': f"Error generating goals: {error_message}"},
            'quote': {'error': f"Error generating quote: {error_message}"}
        }
    
    def format_persona_report(self, username: str, persona: Dict, user_data: Dict, themes: Dict) -> str:
        """Format the persona report for output."""
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
            report_parts.append("")
        
        # Add frustrations
        if 'frustrations' in persona:
            report_parts.append("FRUSTRATIONS:")
            if 'items' in persona['frustrations']:
                for item in persona['frustrations']['items']:
                    report_parts.append(f"- {item}")
            report_parts.append("")
        
        # Add goals
        if 'goals' in persona:
            report_parts.append("GOALS & NEEDS:")
            if 'items' in persona['goals']:
                for item in persona['goals']['items']:
                    report_parts.append(f"- {item}")
            report_parts.append("")
        
        # Add quote
        if 'quote' in persona:
            quote_text = persona['quote']
            if isinstance(quote_text, dict):
                quote_text = quote_text.get('description', '')
            report_parts.extend([
                "REPRESENTATIVE QUOTE:",
                f'"{quote_text}"',
                ""
            ])
        
        # Add statistics
        report_parts.extend([
            "=" * 60,
            "USER STATISTICS:",
            f"- Account Created: {format_timestamp(user_data['created_utc'])}",
            f"- Comment Karma: {user_data['comment_karma']:,}",
            f"- Link Karma: {user_data['link_karma']:,}",
            f"- Total Posts: {len(user_data['posts'])}",
            f"- Total Comments: {len(user_data['comments'])}",
            "",
            "NLP ANALYSIS:",
            f"- Overall Sentiment: {themes.get('sentiment', 0):.3f} (-1 to 1 scale)",
            f"- Text Analyzed: {themes.get('text_length', 0):,} characters",
            f"- Top Keywords: {format_keywords(themes.get('keywords', [])[:10])}",
            ""
        ])
        
        return "\n".join(report_parts) 