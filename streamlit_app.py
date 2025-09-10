"""
Streamlit Football Match Prediction App
A community-ready web application for predicting football match outcomes.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import Dict, List, Optional
import logging
import re
from bs4 import BeautifulSoup
import time as time_module
import os

# Import real scraper and LLM analyzer
try:
    from real_scraper import scraper, llm_analyzer, MatchFixture, NewsArticle
    REAL_SCRAPER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Real scraper not available: {e}")
    REAL_SCRAPER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="⚽ Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .match-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Available leagues
AVAILABLE_LEAGUES = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Champions League"
]

# Sample fixture data (in a real app, this would come from scraping)
SAMPLE_FIXTURES = {
    "2025-09-13": [
        {
            "home_team": "Real Sociedad",
            "away_team": "Real Madrid",
            "match_time": "14:15",
            "league": "La Liga",
            "venue": "Reale Arena"
        },
        {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "match_time": "16:30",
            "league": "Premier League",
            "venue": "Old Trafford"
        },
        {
            "home_team": "Bayern Munich",
            "away_team": "Borussia Dortmund",
            "match_time": "17:30",
            "league": "Bundesliga",
            "venue": "Allianz Arena"
        },
        {
            "home_team": "Juventus",
            "away_team": "AC Milan",
            "match_time": "20:45",
            "league": "Serie A",
            "venue": "Allianz Stadium"
        }
    ],
    "2025-09-14": [
        {
            "home_team": "Barcelona",
            "away_team": "Atletico Madrid",
            "match_time": "16:15",
            "league": "La Liga",
            "venue": "Camp Nou"
        },
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "match_time": "14:30",
            "league": "Premier League",
            "venue": "Emirates Stadium"
        },
        {
            "home_team": "Paris Saint-Germain",
            "away_team": "Marseille",
            "match_time": "20:45",
            "league": "Ligue 1",
            "venue": "Parc des Princes"
        },
        {
            "home_team": "Real Madrid",
            "away_team": "Manchester City",
            "match_time": "21:00",
            "league": "Champions League",
            "venue": "Santiago Bernabéu"
        }
    ],
    "2025-09-15": [
        {
            "home_team": "Liverpool",
            "away_team": "Tottenham",
            "match_time": "16:30",
            "league": "Premier League",
            "venue": "Anfield"
        },
        {
            "home_team": "Inter Milan",
            "away_team": "Napoli",
            "match_time": "20:45",
            "league": "Serie A",
            "venue": "San Siro"
        },
        {
            "home_team": "RB Leipzig",
            "away_team": "Bayer Leverkusen",
            "match_time": "17:30",
            "league": "Bundesliga",
            "venue": "Red Bull Arena"
        },
        {
            "home_team": "Barcelona",
            "away_team": "Bayern Munich",
            "match_time": "21:00",
            "league": "Champions League",
            "venue": "Camp Nou"
        }
    ],
    "2025-09-16": [
        {
            "home_team": "Manchester City",
            "away_team": "Arsenal",
            "match_time": "17:30",
            "league": "Premier League",
            "venue": "Etihad Stadium"
        },
        {
            "home_team": "Sevilla",
            "away_team": "Real Betis",
            "match_time": "21:00",
            "league": "La Liga",
            "venue": "Ramón Sánchez-Pizjuán"
        },
        {
            "home_team": "Lyon",
            "away_team": "Monaco",
            "match_time": "20:45",
            "league": "Ligue 1",
            "venue": "Groupama Stadium"
        }
    ],
    "2025-09-17": [
        {
            "home_team": "Chelsea",
            "away_team": "Newcastle",
            "match_time": "15:00",
            "league": "Premier League",
            "venue": "Stamford Bridge"
        },
        {
            "home_team": "Roma",
            "away_team": "Lazio",
            "match_time": "18:00",
            "league": "Serie A",
            "venue": "Stadio Olimpico"
        },
        {
            "home_team": "Eintracht Frankfurt",
            "away_team": "Wolfsburg",
            "match_time": "15:30",
            "league": "Bundesliga",
            "venue": "Deutsche Bank Park"
        },
        {
            "home_team": "Atletico Madrid",
            "away_team": "Liverpool",
            "match_time": "21:00",
            "league": "Champions League",
            "venue": "Wanda Metropolitano"
        }
    ]
}

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'scraped_fixtures' not in st.session_state:
    st.session_state.scraped_fixtures = {}

if 'last_scrape_time' not in st.session_state:
    st.session_state.last_scrape_time = None

def scrape_real_fixtures(target_date: date, leagues: List[str] = None) -> List[Dict]:
    """Scrape real fixtures using the comprehensive scraper."""
    if not REAL_SCRAPER_AVAILABLE:
        st.warning("Real scraper not available. Using sample data.")
        return []
    
    try:
        # Use the real scraper
        fixtures = scraper.scrape_fixtures_for_date(target_date, leagues)
        
        # Convert MatchFixture objects to dictionaries
        fixture_dicts = []
        for fixture in fixtures:
            fixture_dicts.append({
                'home_team': fixture.home_team,
                'away_team': fixture.away_team,
                'match_time': fixture.match_time,
                'league': fixture.league,
                'venue': fixture.venue
            })
        
        return fixture_dicts
    
    except Exception as e:
        logger.error(f"Error scraping real fixtures: {e}")
        return []

def scrape_news_for_teams(teams: List[str], days_back: int = 7) -> List[Dict]:
    """Scrape news articles for specific teams."""
    if not REAL_SCRAPER_AVAILABLE:
        return []
    
    try:
        # Use the real scraper to get news
        articles = scraper.scrape_news_for_teams(teams, days_back)
        
        # Convert NewsArticle objects to dictionaries
        article_dicts = []
        for article in articles:
            article_dicts.append({
                'title': article.title,
                'content': article.content,
                'url': article.url,
                'published_date': article.published_date,
                'source': article.source,
                'sentiment_score': article.sentiment_score,
                'relevance_score': article.relevance_score
            })
        
        return article_dicts
    
    except Exception as e:
        logger.error(f"Error scraping news: {e}")
        return []

def analyze_news_with_llm(home_team: str, away_team: str, articles: List[Dict]) -> Dict:
    """Analyze news articles using LLM."""
    if not REAL_SCRAPER_AVAILABLE or not articles:
        return {
            "form_analysis": "No news analysis available",
            "key_factors": "Standard match factors",
            "prediction": "Home",
            "confidence": 0.5,
            "risks": "Standard risks"
        }
    
    try:
        # Convert dictionaries back to NewsArticle objects
        news_articles = []
        for article_dict in articles:
            news_articles.append(NewsArticle(
                title=article_dict['title'],
                content=article_dict['content'],
                url=article_dict['url'],
                published_date=article_dict['published_date'],
                source=article_dict['source'],
                sentiment_score=article_dict['sentiment_score'],
                relevance_score=article_dict['relevance_score']
            ))
        
        # Use LLM analyzer
        analysis = llm_analyzer.analyze_news_for_match(home_team, away_team, news_articles)
        return analysis
    
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return {
            "form_analysis": "Analysis failed",
            "key_factors": "Standard match factors",
            "prediction": "Home",
            "confidence": 0.5,
            "risks": "Standard risks"
        }

def get_real_fixtures_for_date(target_date: date):
    """Get real fixtures for a specific date using the comprehensive scraper."""
    date_str = target_date.strftime("%Y-%m-%d")
    
    # Check if we have cached data (less than 1 hour old)
    if (st.session_state.last_scrape_time and 
        datetime.now() - st.session_state.last_scrape_time < timedelta(hours=1) and
        date_str in st.session_state.scraped_fixtures):
        return st.session_state.scraped_fixtures[date_str]
    
    fixtures = []
    
    # Use the real scraper if available
    if REAL_SCRAPER_AVAILABLE:
        try:
            fixtures = scrape_real_fixtures(target_date)
            if fixtures:
                # Cache the results
                st.session_state.scraped_fixtures[date_str] = fixtures
                st.session_state.last_scrape_time = datetime.now()
                return fixtures
        except Exception as e:
            logger.warning(f"Real scraping failed: {e}")
    
    # Fallback to enhanced sample data
    logger.info("Using enhanced sample data as fallback")
    fixtures = get_enhanced_sample_fixtures(target_date)
    
    # Cache the results
    st.session_state.scraped_fixtures[date_str] = fixtures
    st.session_state.last_scrape_time = datetime.now()
    
    return fixtures

def get_enhanced_sample_fixtures(target_date: date):
    """Enhanced sample fixtures with more realistic data."""
    date_str = target_date.strftime("%Y-%m-%d")
    
    # More realistic fixture data based on actual schedules
    enhanced_fixtures = {
        "2025-09-13": [
            {
                "home_team": "Real Sociedad",
                "away_team": "Real Madrid",
                "match_time": "14:15",
                "league": "La Liga",
                "venue": "Reale Arena"
            },
            {
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "match_time": "16:30",
                "league": "Premier League",
                "venue": "Old Trafford"
            },
            {
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "match_time": "17:30",
                "league": "Bundesliga",
                "venue": "Allianz Arena"
            }
        ],
        "2025-09-14": [
            {
                "home_team": "Barcelona",
                "away_team": "Valencia",  # Corrected as per user feedback
                "match_time": "16:15",
                "league": "La Liga",
                "venue": "Camp Nou"
            },
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "match_time": "14:30",
                "league": "Premier League",
                "venue": "Emirates Stadium"
            },
            {
                "home_team": "Real Madrid",
                "away_team": "Manchester City",
                "match_time": "21:00",
                "league": "Champions League",
                "venue": "Santiago Bernabéu"
            }
        ],
        "2025-09-15": [
            {
                "home_team": "Liverpool",
                "away_team": "Tottenham",
                "match_time": "16:30",
                "league": "Premier League",
                "venue": "Anfield"
            },
            {
                "home_team": "Inter Milan",
                "away_team": "Napoli",
                "match_time": "20:45",
                "league": "Serie A",
                "venue": "San Siro"
            },
            {
                "home_team": "Barcelona",
                "away_team": "Bayern Munich",
                "match_time": "21:00",
                "league": "Champions League",
                "venue": "Camp Nou"
            }
        ]
    }
    
    return enhanced_fixtures.get(date_str, [])

def get_prediction_from_api(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Optional[Dict]:
    """Get prediction from the FastAPI backend."""
    try:
        # Prepare request data
        request_data = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "league": league.lower().replace(" ", "_"),
            "include_confidence": True
        }
        
        # Make API request (assuming local API is running)
        # In production, replace with your deployed API URL
        api_url = "http://localhost:8000/predict"
        
        try:
            response = requests.post(api_url, json=request_data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            # Fallback to mock prediction if API is not available
            return get_mock_prediction(home_team, away_team, match_date, match_time, league)
            
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

# Team strength database (in a real app, this would be from a database)
TEAM_STRENGTHS = {
    # Premier League
    "Manchester City": {"elo": 1850, "form": 0.8, "home_advantage": 0.15},
    "Arsenal": {"elo": 1820, "form": 0.75, "home_advantage": 0.12},
    "Liverpool": {"elo": 1800, "form": 0.7, "home_advantage": 0.18},
    "Manchester United": {"elo": 1750, "form": 0.65, "home_advantage": 0.16},
    "Chelsea": {"elo": 1720, "form": 0.6, "home_advantage": 0.14},
    "Tottenham": {"elo": 1700, "form": 0.55, "home_advantage": 0.13},
    "Newcastle United": {"elo": 1680, "form": 0.7, "home_advantage": 0.17},
    "Brighton & Hove Albion": {"elo": 1650, "form": 0.6, "home_advantage": 0.11},
    "Aston Villa": {"elo": 1630, "form": 0.65, "home_advantage": 0.15},
    "West Ham United": {"elo": 1600, "form": 0.5, "home_advantage": 0.12},
    
    # La Liga
    "Real Madrid": {"elo": 1900, "form": 0.85, "home_advantage": 0.2},
    "Barcelona": {"elo": 1880, "form": 0.8, "home_advantage": 0.18},
    "Atletico Madrid": {"elo": 1820, "form": 0.75, "home_advantage": 0.16},
    "Real Sociedad": {"elo": 1750, "form": 0.7, "home_advantage": 0.14},
    "Sevilla": {"elo": 1720, "form": 0.65, "home_advantage": 0.15},
    "Valencia": {"elo": 1680, "form": 0.6, "home_advantage": 0.13},
    "Real Betis": {"elo": 1650, "form": 0.55, "home_advantage": 0.12},
    "Villarreal": {"elo": 1630, "form": 0.6, "home_advantage": 0.14},
    
    # Serie A
    "Inter Milan": {"elo": 1850, "form": 0.8, "home_advantage": 0.16},
    "AC Milan": {"elo": 1820, "form": 0.75, "home_advantage": 0.15},
    "Juventus": {"elo": 1800, "form": 0.7, "home_advantage": 0.17},
    "Napoli": {"elo": 1780, "form": 0.65, "home_advantage": 0.14},
    "Atalanta": {"elo": 1750, "form": 0.6, "home_advantage": 0.13},
    "Roma": {"elo": 1720, "form": 0.55, "home_advantage": 0.15},
    "Lazio": {"elo": 1700, "form": 0.6, "home_advantage": 0.14},
    
    # Bundesliga
    "Bayern Munich": {"elo": 1920, "form": 0.85, "home_advantage": 0.19},
    "Borussia Dortmund": {"elo": 1850, "form": 0.8, "home_advantage": 0.17},
    "RB Leipzig": {"elo": 1800, "form": 0.75, "home_advantage": 0.15},
    "Bayer Leverkusen": {"elo": 1750, "form": 0.7, "home_advantage": 0.14},
    "Eintracht Frankfurt": {"elo": 1720, "form": 0.65, "home_advantage": 0.13},
    "Union Berlin": {"elo": 1700, "form": 0.6, "home_advantage": 0.12},
    
    # Ligue 1
    "Paris Saint-Germain": {"elo": 1900, "form": 0.85, "home_advantage": 0.18},
    "Marseille": {"elo": 1750, "form": 0.7, "home_advantage": 0.15},
    "Monaco": {"elo": 1720, "form": 0.65, "home_advantage": 0.14},
    "Lens": {"elo": 1700, "form": 0.6, "home_advantage": 0.13},
    "Lyon": {"elo": 1680, "form": 0.55, "home_advantage": 0.12},
    
    # Champions League (same teams, higher stakes)
    "Manchester City": {"elo": 1950, "form": 0.9, "home_advantage": 0.2},
    "Real Madrid": {"elo": 1950, "form": 0.9, "home_advantage": 0.22},
    "Barcelona": {"elo": 1920, "form": 0.85, "home_advantage": 0.2},
    "Bayern Munich": {"elo": 1950, "form": 0.9, "home_advantage": 0.21},
    "Liverpool": {"elo": 1900, "form": 0.8, "home_advantage": 0.19},
    "Manchester United": {"elo": 1850, "form": 0.75, "home_advantage": 0.18},
}

def calculate_elo_probability(home_elo: float, away_elo: float, home_advantage: float = 0.1) -> Dict[str, float]:
    """Calculate match probabilities using Elo ratings."""
    # Adjust home team's effective rating with home advantage
    effective_home_elo = home_elo + (home_advantage * 100)
    
    # Calculate expected score for home team
    expected_home = 1 / (1 + 10 ** ((away_elo - effective_home_elo) / 400))
    
    # Convert to match outcome probabilities
    # Home win probability
    home_win_prob = expected_home * 0.7  # Most expected score becomes home win
    
    # Draw probability (based on Elo difference)
    elo_diff = abs(effective_home_elo - away_elo)
    if elo_diff < 50:
        draw_prob = 0.3  # Close teams more likely to draw
    elif elo_diff < 100:
        draw_prob = 0.25
    else:
        draw_prob = 0.2
    
    # Away win probability
    away_win_prob = 1 - home_win_prob - draw_prob
    
    # Ensure probabilities are reasonable
    home_win_prob = max(0.1, min(0.8, home_win_prob))
    away_win_prob = max(0.1, min(0.8, away_win_prob))
    draw_prob = max(0.1, min(0.4, draw_prob))
    
    # Normalize
    total = home_win_prob + draw_prob + away_win_prob
    return {
        "home_win": home_win_prob / total,
        "draw": draw_prob / total,
        "away_win": away_win_prob / total
    }

def calculate_form_factor(team: str, league: str) -> float:
    """Calculate form factor based on recent performance."""
    team_data = TEAM_STRENGTHS.get(team, {"form": 0.5, "elo": 1500})
    base_form = team_data["form"]
    
    # Add some randomness to simulate recent form variations
    np.random.seed(hash(team + league + str(datetime.now().day)) % 2**32)
    form_variation = np.random.normal(0, 0.1)
    
    return max(0.1, min(0.9, base_form + form_variation))

def calculate_head_to_head_factor(home_team: str, away_team: str) -> float:
    """Calculate head-to-head factor (simplified)."""
    # In a real system, this would use historical H2H data
    # For now, we'll use a simple factor based on team names
    h2h_seed = hash(home_team + away_team) % 100
    if h2h_seed < 40:
        return 0.1  # Home team advantage
    elif h2h_seed < 60:
        return 0.0  # Neutral
    else:
        return -0.1  # Away team advantage

def get_advanced_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Generate advanced prediction using multiple algorithms."""
    
    # Get team data
    home_data = TEAM_STRENGTHS.get(home_team, {"elo": 1500, "form": 0.5, "home_advantage": 0.1})
    away_data = TEAM_STRENGTHS.get(away_team, {"elo": 1500, "form": 0.5, "home_advantage": 0.1})
    
    # Calculate base Elo probabilities
    elo_probs = calculate_elo_probability(
        home_data["elo"], 
        away_data["elo"], 
        home_data["home_advantage"]
    )
    
    # Calculate form factors
    home_form = calculate_form_factor(home_team, league)
    away_form = calculate_form_factor(away_team, league)
    
    # Calculate head-to-head factor
    h2h_factor = calculate_head_to_head_factor(home_team, away_team)
    
    # Adjust probabilities based on form
    form_adjustment = (home_form - away_form) * 0.2
    
    # Final probabilities with adjustments
    home_win_prob = elo_probs["home_win"] + form_adjustment + h2h_factor
    away_win_prob = elo_probs["away_win"] - form_adjustment - h2h_factor
    draw_prob = elo_probs["draw"]
    
    # Ensure probabilities are valid
    home_win_prob = max(0.05, min(0.85, home_win_prob))
    away_win_prob = max(0.05, min(0.85, away_win_prob))
    draw_prob = max(0.1, min(0.4, draw_prob))
    
    # Normalize
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total
    draw_prob /= total
    away_win_prob /= total
    
    # Determine predicted outcome
    probs = [home_win_prob, draw_prob, away_win_prob]
    predicted_outcome = ['Home', 'Draw', 'Away'][np.argmax(probs)]
    confidence = max(probs)
    
    # Calculate model confidence based on Elo difference
    elo_diff = abs(home_data["elo"] - away_data["elo"])
    if elo_diff > 200:
        confidence = min(0.9, confidence + 0.1)  # High confidence for big differences
    elif elo_diff < 50:
        confidence = max(0.3, confidence - 0.1)  # Lower confidence for close matches
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "match_time": match_time,
        "league": league,
        "predictions": {
            "home_win": round(home_win_prob, 3),
            "draw": round(draw_prob, 3),
            "away_win": round(away_win_prob, 3)
        },
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 3),
        "model_used": "Advanced Elo + Form + H2H Model",
        "timestamp": datetime.now().isoformat(),
        "model_details": {
            "home_elo": home_data["elo"],
            "away_elo": away_data["elo"],
            "home_form": round(home_form, 2),
            "away_form": round(away_form, 2),
            "elo_difference": round(abs(home_data["elo"] - away_data["elo"]), 0)
        }
    }

def get_poisson_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Generate prediction using Poisson distribution model."""
    home_data = TEAM_STRENGTHS.get(home_team, {"elo": 1500, "form": 0.5, "home_advantage": 0.1})
    away_data = TEAM_STRENGTHS.get(away_team, {"elo": 1500, "form": 0.5, "home_advantage": 0.1})
    
    # Calculate expected goals based on Elo ratings
    home_expected_goals = (home_data["elo"] / 1000) * home_data["form"] * (1 + home_data["home_advantage"])
    away_expected_goals = (away_data["elo"] / 1000) * away_data["form"]
    
    # Normalize to realistic goal expectations
    home_expected_goals = max(0.5, min(3.0, home_expected_goals))
    away_expected_goals = max(0.5, min(3.0, away_expected_goals))
    
    # Calculate probabilities using Poisson distribution
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    # Calculate probabilities for different scorelines
    for home_goals in range(6):  # 0-5 goals
        for away_goals in range(6):
            # Poisson probability for this scoreline
            prob = (np.exp(-home_expected_goals) * (home_expected_goals ** home_goals) / 
                   np.math.factorial(home_goals)) * \
                   (np.exp(-away_expected_goals) * (away_expected_goals ** away_goals) / 
                   np.math.factorial(away_goals))
            
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob
    
    # Normalize probabilities
    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
    
    # Determine predicted outcome
    probs = [home_win_prob, draw_prob, away_win_prob]
    predicted_outcome = ['Home', 'Draw', 'Away'][np.argmax(probs)]
    confidence = max(probs)
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "match_time": match_time,
        "league": league,
        "predictions": {
            "home_win": round(home_win_prob, 3),
            "draw": round(draw_prob, 3),
            "away_win": round(away_win_prob, 3)
        },
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 3),
        "model_used": "Poisson Distribution Model",
        "timestamp": datetime.now().isoformat(),
        "model_details": {
            "home_expected_goals": round(home_expected_goals, 2),
            "away_expected_goals": round(away_expected_goals, 2)
        }
    }

def get_ensemble_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Generate ensemble prediction combining multiple models and LLM news analysis."""
    
    # Get predictions from different models
    elo_pred = get_advanced_prediction(home_team, away_team, match_date, match_time, league)
    poisson_pred = get_poisson_prediction(home_team, away_team, match_date, match_time, league)
    
    # Get LLM news analysis
    news_analysis = get_llm_enhanced_prediction(home_team, away_team, match_date, match_time, league)
    
    # Weight the models (Elo gets more weight, LLM analysis adds intelligence)
    elo_weight = 0.5
    poisson_weight = 0.3
    llm_weight = 0.2
    
    # Combine predictions
    combined_home = (elo_pred["predictions"]["home_win"] * elo_weight + 
                    poisson_pred["predictions"]["home_win"] * poisson_weight +
                    news_analysis["predictions"]["home_win"] * llm_weight)
    combined_draw = (elo_pred["predictions"]["draw"] * elo_weight + 
                    poisson_pred["predictions"]["draw"] * poisson_weight +
                    news_analysis["predictions"]["draw"] * llm_weight)
    combined_away = (elo_pred["predictions"]["away_win"] * elo_weight + 
                    poisson_pred["predictions"]["away_win"] * poisson_weight +
                    news_analysis["predictions"]["away_win"] * llm_weight)
    
    # Normalize
    total = combined_home + combined_draw + combined_away
    combined_home /= total
    combined_draw /= total
    combined_away /= total
    
    # Determine predicted outcome
    probs = [combined_home, combined_draw, combined_away]
    predicted_outcome = ['Home', 'Draw', 'Away'][np.argmax(probs)]
    confidence = max(probs)
    
    # Combine model details
    combined_details = elo_pred.get("model_details", {}).copy()
    combined_details.update(poisson_pred.get("model_details", {}))
    combined_details.update(news_analysis.get("model_details", {}))
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "match_time": match_time,
        "league": league,
        "predictions": {
            "home_win": round(combined_home, 3),
            "draw": round(combined_draw, 3),
            "away_win": round(combined_away, 3)
        },
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 3),
        "model_used": "Ensemble (Elo + Poisson + LLM News)",
        "timestamp": datetime.now().isoformat(),
        "model_details": combined_details,
        "news_analysis": news_analysis.get("news_analysis", {})
    }

def get_llm_enhanced_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Get LLM-enhanced prediction based on news analysis."""
    
    # Scrape news for the teams
    teams = [home_team, away_team]
    articles = scrape_news_for_teams(teams, days_back=7)
    
    # Analyze news with LLM
    news_analysis = analyze_news_with_llm(home_team, away_team, articles)
    
    # Convert LLM analysis to prediction probabilities
    # This is a simplified conversion - in practice, you'd want more sophisticated logic
    llm_prediction = news_analysis.get("prediction", "Home")
    llm_confidence = news_analysis.get("confidence", 0.5)
    
    # Convert to probabilities
    if llm_prediction == "Home":
        home_win = llm_confidence
        draw = (1 - llm_confidence) * 0.3
        away_win = (1 - llm_confidence) * 0.7
    elif llm_prediction == "Away":
        away_win = llm_confidence
        draw = (1 - llm_confidence) * 0.3
        home_win = (1 - llm_confidence) * 0.7
    else:  # Draw
        draw = llm_confidence
        home_win = (1 - llm_confidence) * 0.5
        away_win = (1 - llm_confidence) * 0.5
    
    # Normalize
    total = home_win + draw + away_win
    home_win /= total
    draw /= total
    away_win /= total
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "match_time": match_time,
        "league": league,
        "predictions": {
            "home_win": round(home_win, 3),
            "draw": round(draw, 3),
            "away_win": round(away_win, 3)
        },
        "predicted_outcome": llm_prediction,
        "confidence": round(llm_confidence, 3),
        "model_used": "LLM News Analysis",
        "timestamp": datetime.now().isoformat(),
        "model_details": {
            "articles_analyzed": len(articles),
            "avg_sentiment": sum(a.get("sentiment_score", 0) for a in articles) / max(len(articles), 1),
            "news_confidence": llm_confidence
        },
        "news_analysis": news_analysis
    }

def get_mock_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Generate advanced prediction using ensemble of sophisticated algorithms."""
    return get_ensemble_prediction(home_team, away_team, match_date, match_time, league)

def display_prediction_result(prediction: Dict, chart_key: str = ""):
    """Display prediction results in a nice format."""
    if not prediction:
        return
    
    # Main prediction card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    # Match info
    match_time = prediction.get('match_time', 'TBD')
    league = prediction.get('league', 'Unknown')
    venue = prediction.get('venue', 'TBD')
    model_details = prediction.get('model_details', {})
    
    st.markdown(f"""
    <div class="match-info">
        <div class="team-name">{prediction['home_team']} vs {prediction['away_team']}</div>
        <p><strong>Date:</strong> {prediction['match_date']} | <strong>Time:</strong> {match_time}</p>
        <p><strong>League:</strong> {league} | <strong>Venue:</strong> {venue}</p>
        <p><strong>Predicted Outcome:</strong> {prediction['predicted_outcome']}</p>
        <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
        <p><strong>Model Used:</strong> {prediction['model_used']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model details if available
    if model_details:
        st.markdown("#### 📊 Model Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Home Team Elo", f"{model_details.get('home_elo', 'N/A')}")
            st.metric("Home Form", f"{model_details.get('home_form', 'N/A'):.2f}")
        
        with col2:
            st.metric("Away Team Elo", f"{model_details.get('away_elo', 'N/A')}")
            st.metric("Away Form", f"{model_details.get('away_form', 'N/A'):.2f}")
        
        with col3:
            st.metric("Elo Difference", f"{model_details.get('elo_difference', 'N/A')}")
            if model_details.get('elo_difference', 0) > 100:
                st.success("Strong favorite")
            elif model_details.get('elo_difference', 0) < 50:
                st.warning("Close match")
            else:
                st.info("Moderate favorite")
    
    # Show news analysis if available
    news_analysis = prediction.get('news_analysis', {})
    if news_analysis and news_analysis.get('form_analysis'):
        st.markdown("#### 📰 News Analysis")
        
        # Form analysis
        st.markdown("**Recent Form & News:**")
        st.info(news_analysis.get('form_analysis', 'No analysis available'))
        
        # Key factors
        if news_analysis.get('key_factors'):
            st.markdown("**Key Factors:**")
            st.warning(news_analysis.get('key_factors', 'Standard factors'))
        
        # Risks
        if news_analysis.get('risks'):
            st.markdown("**Risk Factors:**")
            st.error(news_analysis.get('risks', 'Standard risks'))
        
        # News metrics
        if model_details.get('articles_analyzed', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Articles Analyzed", model_details.get('articles_analyzed', 0))
            with col2:
                sentiment = model_details.get('avg_sentiment', 0)
                st.metric("Avg Sentiment", f"{sentiment:.2f}")
                if sentiment > 0.1:
                    st.success("Positive sentiment")
                elif sentiment < -0.1:
                    st.error("Negative sentiment")
                else:
                    st.info("Neutral sentiment")
            with col3:
                st.metric("News Confidence", f"{model_details.get('news_confidence', 0):.2f}")
    
    # Probability breakdown
    probs = prediction['predictions']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏠 Home Win",
            value=f"{probs['home_win']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🤝 Draw",
            value=f"{probs['draw']:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="✈️ Away Win",
            value=f"{probs['away_win']:.1%}",
            delta=None
        )
    
    # Probability visualization
    fig = go.Figure(data=[
        go.Bar(
            x=['Home Win', 'Draw', 'Away Win'],
            y=[probs['home_win'], probs['draw'], probs['away_win']],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f"{probs['home_win']:.1%}", f"{probs['draw']:.1%}", f"{probs['away_win']:.1%}"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Outcome",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"prob_chart_{chart_key}")
    
    # Confidence indicator
    confidence = prediction['confidence']
    if confidence >= 0.6:
        confidence_class = "confidence-high"
        confidence_text = "High Confidence"
    elif confidence >= 0.4:
        confidence_class = "confidence-medium"
        confidence_text = "Medium Confidence"
    else:
        confidence_class = "confidence-low"
        confidence_text = "Low Confidence"
    
    st.markdown(f'<p class="{confidence_class}">🎯 {confidence_text}</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_prediction_history():
    """Display prediction history."""
    if not st.session_state.predictions_history:
        st.info("No predictions made yet. Make your first prediction above!")
        return
    
    st.subheader("📊 Prediction History")
    
    # Convert to DataFrame for better display
    history_df = pd.DataFrame(st.session_state.predictions_history)
    
    # Display recent predictions
    for i, pred in enumerate(reversed(st.session_state.predictions_history[-5:])):
        match_id = len(st.session_state.predictions_history) - i
        with st.expander(f"Match {match_id}: {pred['home_team']} vs {pred['away_team']}"):
            display_prediction_result(pred, chart_key=f"history_{match_id}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Predict outcomes for scheduled football matches using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for date selection
    with st.sidebar:
        st.header("📅 Select Match Date")
        
        # Date selection
        selected_date = st.date_input(
            "Choose Date",
            value=date(2025, 9, 13),  # Default to a date with matches
            min_value=date(2025, 9, 13),
            max_value=date(2025, 9, 17)
        )
        
        # League filter
        st.subheader("🏆 Filter by League")
        selected_leagues = st.multiselect(
            "Select Leagues",
            options=AVAILABLE_LEAGUES,
            default=AVAILABLE_LEAGUES
        )
        
        # Model selection
        st.subheader("🤖 Prediction Model")
        model_choice = st.selectbox(
            "Choose Prediction Model",
            options=[
                "Ensemble (Recommended)",
                "LLM News Analysis",
                "Elo + Form + H2H",
                "Poisson Distribution",
                "Simple Random"
            ],
            help="Ensemble model combines multiple algorithms including LLM news analysis for best accuracy"
        )
        
        # Additional options
        st.subheader("⚙️ Options")
        show_confidence = st.checkbox("Show Confidence Analysis", value=True)
        save_prediction = st.checkbox("Save to History", value=True)
        
        # Refresh button
        st.subheader("🔄 Data Refresh")
        if st.button("🔄 Refresh Match Data", help="Fetch latest match fixtures from web sources"):
            # Clear cached data
            st.session_state.scraped_fixtures = {}
            st.session_state.last_scrape_time = None
            st.success("Match data refreshed! Select a date to see updated fixtures.")
            st.rerun()
    
    # Main content area
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Get matches for selected date using real-time scraping
    matches = get_real_fixtures_for_date(selected_date)
    
    # Filter by selected leagues
    if selected_leagues:
        matches = [match for match in matches if match['league'] in selected_leagues]
    
    if matches:
        st.subheader(f"🏟️ Matches on {selected_date.strftime('%B %d, %Y')}")
        
        # Group matches by league
        matches_by_league = {}
        for match in matches:
            league = match['league']
            if league not in matches_by_league:
                matches_by_league[league] = []
            matches_by_league[league].append(match)
        
        # Display matches by league
        for league, league_matches in matches_by_league.items():
            st.markdown(f"### {league}")
            
            for i, match in enumerate(league_matches):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="match-info">
                            <div class="team-name">{match['home_team']} vs {match['away_team']}</div>
                            <p><strong>Time:</strong> {match['match_time']} | <strong>Venue:</strong> {match['venue']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button(f"🔮 Predict", key=f"predict_{league}_{i}"):
                            with st.spinner("Generating prediction..."):
                                # Choose prediction method based on user selection
                                if model_choice == "Ensemble (Recommended)":
                                    prediction = get_ensemble_prediction(
                                        match['home_team'], 
                                        match['away_team'], 
                                        date_str, 
                                        match['match_time'], 
                                        match['league']
                                    )
                                elif model_choice == "LLM News Analysis":
                                    prediction = get_llm_enhanced_prediction(
                                        match['home_team'], 
                                        match['away_team'], 
                                        date_str, 
                                        match['match_time'], 
                                        match['league']
                                    )
                                elif model_choice == "Elo + Form + H2H":
                                    prediction = get_advanced_prediction(
                                        match['home_team'], 
                                        match['away_team'], 
                                        date_str, 
                                        match['match_time'], 
                                        match['league']
                                    )
                                elif model_choice == "Poisson Distribution":
                                    prediction = get_poisson_prediction(
                                        match['home_team'], 
                                        match['away_team'], 
                                        date_str, 
                                        match['match_time'], 
                                        match['league']
                                    )
                                else:  # Simple Random
                                    prediction = get_prediction_from_api(
                                        match['home_team'], 
                                        match['away_team'], 
                                        date_str, 
                                        match['match_time'], 
                                        match['league']
                                    )
                                
                                if prediction:
                                    # Add venue to prediction
                                    prediction['venue'] = match['venue']
                                    
                                    # Display prediction
                                    st.success("✅ Prediction generated!")
                                    display_prediction_result(prediction, chart_key=f"match_{league}_{i}")
                                    
                                    # Save to history if requested
                                    if save_prediction:
                                        st.session_state.predictions_history.append(prediction)
                                        st.success("💾 Saved to history!")
                                    
                                    # Confidence analysis
                                    if show_confidence:
                                        st.subheader("📈 Confidence Analysis")
                                        
                                        confidence = prediction['confidence']
                                        probs = prediction['predictions']
                                        
                                        # Create confidence gauge
                                        fig_gauge = go.Figure(go.Indicator(
                                            mode = "gauge+number+delta",
                                            value = confidence * 100,
                                            domain = {'x': [0, 1], 'y': [0, 1]},
                                            title = {'text': "Prediction Confidence (%)"},
                                            delta = {'reference': 50},
                                            gauge = {
                                                'axis': {'range': [None, 100]},
                                                'bar': {'color': "darkblue"},
                                                'steps': [
                                                    {'range': [0, 40], 'color': "lightgray"},
                                                    {'range': [40, 70], 'color': "yellow"},
                                                    {'range': [70, 100], 'color': "green"}
                                                ],
                                                'threshold': {
                                                    'line': {'color': "red", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': 90
                                                }
                                            }
                                        ))
                                        
                                        fig_gauge.update_layout(height=300)
                                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"confidence_gauge_{league}_{i}")
                                        
                                        # Probability distribution pie chart
                                        fig_pie = px.pie(
                                            values=list(probs.values()),
                                            names=list(probs.keys()),
                                            title="Outcome Probability Distribution",
                                            color_discrete_map={
                                                'home_win': '#1f77b4',
                                                'draw': '#ff7f0e',
                                                'away_win': '#2ca02c'
                                            }
                                        )
                                        st.plotly_chart(fig_pie, use_container_width=True, key=f"probability_pie_{league}_{i}")
                    
                    with col3:
                        # Quick prediction preview
                        if st.button(f"👁️ Preview", key=f"preview_{league}_{i}"):
                            # Generate quick preview using selected model
                            if model_choice == "Ensemble (Recommended)":
                                preview_pred = get_ensemble_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                            elif model_choice == "LLM News Analysis":
                                preview_pred = get_llm_enhanced_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                            elif model_choice == "Elo + Form + H2H":
                                preview_pred = get_advanced_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                            elif model_choice == "Poisson Distribution":
                                preview_pred = get_poisson_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                            else:  # Simple Random
                                preview_pred = get_mock_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                            
                            preview_pred['venue'] = match['venue']
                            
                            # Show quick preview with model info
                            st.info(f"Quick Preview ({model_choice}): {preview_pred['predicted_outcome']} ({preview_pred['confidence']:.1%})")
                    
                    st.markdown("---")
    else:
        st.info(f"No matches found for {selected_date.strftime('%B %d, %Y')} in the selected leagues.")
    
    # Display prediction history
    display_prediction_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>⚽ Football Match Predictor | Powered by Machine Learning</p>
        <p>Built with Streamlit | Deploy on Streamlit Community Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
