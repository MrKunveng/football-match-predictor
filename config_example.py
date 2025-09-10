"""
Configuration example for the Football Prediction App.
Copy this file to config.py and fill in your actual API keys.
"""

# OpenAI API Key for LLM News Analysis
OPENAI_API_KEY = "your_openai_api_key_here"

# Alternative LLM APIs (optional)
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Football Data APIs (optional)
FOOTBALL_DATA_API_KEY = "your_football_data_api_key_here"
API_FOOTBALL_KEY = "your_api_football_key_here"

# News API Keys (optional)
NEWS_API_KEY = "your_news_api_key_here"

# Scraping Configuration
SCRAPING_DELAY = 1.0  # Delay between requests in seconds
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# News Sources Configuration
NEWS_SOURCES = {
    'BBC Sport': 'http://feeds.bbci.co.uk/sport/football/rss.xml',
    'ESPN FC': 'https://www.espn.com/espn/rss/soccer/news',
    'Sky Sports': 'https://www.skysports.com/rss/0,20514,11661,00.xml',
    'The Guardian': 'https://www.theguardian.com/football/rss',
    'Goal.com': 'https://www.goal.com/en/feeds/news',
    'Football365': 'https://www.football365.com/feed/',
}

# Fixture Sources Configuration
FIXTURE_SOURCES = {
    'SofaScore': 'https://www.sofascore.com',
    'FotMob': 'https://www.fotmob.com',
    'ESPN': 'https://www.espn.com/soccer/fixtures',
    'BBC Sport': 'https://www.bbc.com/sport/football/scores-fixtures',
}
