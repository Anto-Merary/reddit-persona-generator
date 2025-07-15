#!/usr/bin/env python3
"""
Example usage of the Reddit Persona Generator

This script demonstrates various ways to use the modular persona generator
for different analysis scenarios.
"""

import os
from dotenv import load_dotenv
from persona_generator import PersonaGenerator


def example_basic_usage():
    """Example of basic usage with a single user."""
    print("=== Basic Usage Example ===")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the generator (uses config automatically)
    generator = PersonaGenerator()
    
    # Generate persona for a single user
    profile_url = "https://www.reddit.com/user/kojied/"
    result = generator.generate_persona_from_url(profile_url, "kojied_persona.txt")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Generated persona report for {profile_url}")
        print(f"Report saved to: {result['output_file']}")
    print()


def example_batch_analysis():
    """Example of analyzing multiple users in batch."""
    print("=== Batch Analysis Example ===")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the generator (uses config automatically)
    generator = PersonaGenerator()
    
    # List of users to analyze
    users = [
        "https://www.reddit.com/user/kojied/",
        "https://www.reddit.com/user/Hungry-Move-6603/"
    ]
    
    results = []
    
    for i, user_url in enumerate(users, 1):
        print(f"Processing user {i}/{len(users)}: {user_url}")
        
        try:
            # Extract username for filename
            username = generator.extract_username_from_url(user_url)
            output_file = f"persona_{username}.txt"
            
            # Generate persona
            result = generator.generate_persona_from_url(user_url, output_file)
            
            if 'error' in result:
                raise Exception(result['error'])
            
            results.append({
                'username': username,
                'url': user_url,
                'output_file': result['output_file'],
                'status': 'success'
            })
            
            print(f"✓ Successfully generated persona for {username}")
            
        except Exception as e:
            print(f"✗ Error processing {user_url}: {e}")
            results.append({
                'username': 'unknown',
                'url': user_url,
                'output_file': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Print summary
    print("\n=== Batch Analysis Summary ===")
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"Successfully processed: {len(successful)} users")
    print(f"Failed to process: {len(failed)} users")
    
    if successful:
        print("\nSuccessful analyses:")
        for result in successful:
            print(f"- {result['username']}: {result['output_file']}")
    
    if failed:
        print("\nFailed analyses:")
        for result in failed:
            print(f"- {result['url']}: {result.get('error', 'Unknown error')}")
    
    print()


def example_custom_analysis():
    """Example of custom analysis with specific focus areas."""
    print("=== Custom Analysis Example ===")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the generator (uses config automatically)
    generator = PersonaGenerator()
    
    # Analyze a specific user
    profile_url = "https://www.reddit.com/user/kojied/"
    
    try:
        # Extract username
        username = generator.extract_username_from_url(profile_url)
        
        # Fetch user data
        user_data = generator.fetch_user_data(username)
        
        # Clean and analyze text
        all_texts = []
        for post in user_data['posts']:
            if post['title']:
                all_texts.append(generator.clean_text(post['title']))
            if post['body']:
                all_texts.append(generator.clean_text(post['body']))
        
        for comment in user_data['comments']:
            if comment['body']:
                all_texts.append(generator.clean_text(comment['body']))
        
        # Extract themes
        themes = generator.extract_themes_with_spacy(all_texts)
        
        # Custom analysis
        print(f"Custom Analysis for {username}:")
        print(f"- Total posts: {len(user_data['posts'])}")
        print(f"- Total comments: {len(user_data['comments'])}")
        print(f"- Overall sentiment: {themes.get('sentiment', 0):.3f}")
        print(f"- Most active subreddits:")
        
        # Analyze subreddit activity
        subreddit_counts = {}
        for post in user_data['posts']:
            subreddit = post['subreddit']
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        
        for comment in user_data['comments']:
            subreddit = comment['subreddit']
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        
        # Sort by activity
        sorted_subreddits = sorted(subreddit_counts.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        for subreddit, count in sorted_subreddits[:5]:
            print(f"  - r/{subreddit}: {count} interactions")
        
        print(f"- Top keywords: {[word for word, _ in themes.get('keywords', [])[:5]]}")
        
    except Exception as e:
        print(f"Error in custom analysis: {e}")
    
    print()


def example_error_handling():
    """Example of proper error handling."""
    print("=== Error Handling Example ===")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the generator (uses config automatically)
    generator = PersonaGenerator()
    
    # Test various error scenarios
    test_cases = [
        "https://www.reddit.com/user/nonexistent_user_12345/",
        "https://www.reddit.com/user/private_user_test/",
        "https://www.google.com/user/test",  # Invalid URL
        "not_a_url_at_all",
        "https://www.reddit.com/user/kojied/"  # Valid user
    ]
    
    for test_url in test_cases:
        print(f"\nTesting: {test_url}")
        
        try:
            # Try to extract username
            username = generator.extract_username_from_url(test_url)
            print(f"✓ Username extracted: {username}")
            
            # Try to generate persona
            result = generator.generate_persona_from_url(test_url)
            
            if 'error' in result:
                print(f"⚠ Generated report with errors: {result['error'][:100]}...")
            else:
                print("✓ Successfully generated persona report")
                
        except ValueError as e:
            print(f"✗ URL parsing error: {e}")
        except Exception as e:
            print(f"✗ General error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("Reddit Persona Generator - Usage Examples")
    print("=" * 60)
    
    # Check if credentials are available
    load_dotenv()
    if not all([
        os.getenv('REDDIT_CLIENT_ID'),
        os.getenv('REDDIT_CLIENT_SECRET')
    ]):
        print("⚠ Warning: Reddit API credentials not found in .env file")
        print("Some examples may not work without valid credentials")
        print("Please set up your .env file with valid Reddit API credentials")
        print()
    
    # Run examples
    example_basic_usage()
    example_batch_analysis()
    example_custom_analysis()
    example_error_handling()
    
    print("All examples completed!")


if __name__ == "__main__":
    main() 