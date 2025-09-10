"""
Real web scraper for football fixtures and news analysis.
Uses multiple sources for comprehensive data collection.
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date
import time
import re
import json
from typing import Dict, List, Optional, Tuple
import logging
from fake_useragent import UserAgent
import feedparser
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai
import os
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Try to import configuration
try:
    from config import OPENAI_API_KEY, NEWS_SOURCES, FIXTURE_SOURCES
except ImportError:
    # Use environment variables as fallback
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    NEWS_SOURCES = {
        'BBC Sport': 'http://feeds.bbci.co.uk/sport/football/rss.xml',
        'ESPN FC': 'https://www.espn.com/espn/rss/soccer/news',
        'Sky Sports': 'https://www.skysports.com/rss/0,20514,11661,00.xml',
        'The Guardian': 'https://www.theguardian.com/football/rss',
        'Goal.com': 'https://www.goal.com/en/feeds/news',
        'Football365': 'https://www.football365.com/feed/',
    }
    FIXTURE_SOURCES = {
        'SofaScore': 'https://www.sofascore.com',
        'FotMob': 'https://www.fotmob.com',
        'ESPN': 'https://www.espn.com/soccer/fixtures',
        'BBC Sport': 'https://www.bbc.com/sport/football/scores-fixtures',
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchFixture:
    """Data class for match fixtures."""
    home_team: str
    away_team: str
    match_date: str
    match_time: str
    league: str
    venue: str
    competition: str = ""

@dataclass
class NewsArticle:
    """Data class for news articles."""
    title: str
    content: str
    url: str
    published_date: str
    source: str
    sentiment_score: float
    relevance_score: float

class RealFootballScraper:
    """Real web scraper for football fixtures and news."""
    
    def __init__(self, use_selenium: bool = True):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.use_selenium = use_selenium
        self.driver = None
        
        # News sources
        self.news_sources = NEWS_SOURCES
        
        # Fixture sources
        self.fixture_sources = FIXTURE_SOURCES
        
        # Setup Selenium if needed
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.ua.random}')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to setup Selenium: {e}")
            self.use_selenium = False
    
    def _get_page_selenium(self, url: str) -> BeautifulSoup:
        """Get page content using Selenium."""
        if not self.driver:
            raise RuntimeError("Selenium driver not initialized")
        
        try:
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content to load
            time.sleep(3)
            
            return BeautifulSoup(self.driver.page_source, 'html.parser')
        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            raise
        except WebDriverException as e:
            logger.error(f"Selenium error: {e}")
            raise
    
    def scrape_fixtures_for_date(self, target_date: date, leagues: List[str] = None) -> List[MatchFixture]:
        """Scrape fixtures for a specific date from multiple sources."""
        if leagues is None:
            leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Champions League']
        
        all_fixtures = []
        
        # Try multiple sources
        for source_name, source_url in self.fixture_sources.items():
            try:
                logger.info(f"Scraping fixtures from {source_name}")
                fixtures = self._scrape_source_fixtures(source_url, target_date, leagues)
                all_fixtures.extend(fixtures)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to scrape {source_name}: {e}")
                continue
        
        # Remove duplicates and return
        unique_fixtures = self._deduplicate_fixtures(all_fixtures)
        logger.info(f"Found {len(unique_fixtures)} unique fixtures for {target_date}")
        return unique_fixtures
    
    def _scrape_source_fixtures(self, source_url: str, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Scrape fixtures from a specific source."""
        fixtures = []
        
        try:
            # Use Selenium for dynamic sites like SofaScore and FotMob
            if self.use_selenium and self.driver and ('sofascore.com' in source_url or 'fotmob.com' in source_url):
                soup = self._get_page_selenium(source_url)
            else:
                # Use requests for static sites
                headers = {
                    'User-Agent': self.ua.random,
                    'Referer': 'https://www.google.com/',
                }
                
                response = self.session.get(source_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
            
            # SofaScore scraping
            if 'sofascore.com' in source_url:
                fixtures = self._parse_sofascore_fixtures(soup, target_date, leagues)
            
            # FotMob scraping
            elif 'fotmob.com' in source_url:
                fixtures = self._parse_fotmob_fixtures(soup, target_date, leagues)
            
            # ESPN scraping
            elif 'espn.com' in source_url:
                fixtures = self._parse_espn_fixtures(soup, target_date, leagues)
            
            # BBC Sport scraping
            elif 'bbc.com' in source_url:
                fixtures = self._parse_bbc_fixtures(soup, target_date, leagues)
            
        except Exception as e:
            logger.error(f"Error scraping {source_url}: {e}")
        
        return fixtures
    
    def _parse_sofascore_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse SofaScore fixtures."""
        fixtures = []
        
        try:
            # SofaScore uses specific class names for matches
            match_containers = soup.find_all(['div', 'tr'], class_=re.compile(r'match|fixture|event'))
            
            for container in match_containers:
                try:
                    # Extract team names from SofaScore structure
                    team_elements = container.find_all(['span', 'a'], class_=re.compile(r'team|participant'))
                    if len(team_elements) >= 2:
                        home_team = self._clean_team_name(team_elements[0].get_text(strip=True))
                        away_team = self._clean_team_name(team_elements[1].get_text(strip=True))
                    
                    # Extract time
                    time_elem = container.find(['span', 'div'], class_=re.compile(r'time|date|start'))
                    match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                    
                    # Extract league/tournament
                    league_elem = container.find(['span', 'div'], class_=re.compile(r'tournament|league|competition'))
                    league = league_elem.get_text(strip=True) if league_elem else "Unknown"
                    
                    # Check if league is in our target leagues
                    if any(target_league.lower() in league.lower() for target_league in leagues):
                        fixture = MatchFixture(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=target_date.strftime('%Y-%m-%d'),
                            match_time=match_time,
                            league=league,
                            venue="TBD"
                        )
                        fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing SofaScore fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing SofaScore fixtures: {e}")
        
        return fixtures
    
    def _parse_fotmob_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse FotMob fixtures."""
        fixtures = []
        
        try:
            # FotMob uses specific class names for matches
            match_containers = soup.find_all(['div', 'li'], class_=re.compile(r'match|fixture|game'))
            
            for container in match_containers:
                try:
                    # Extract team names from FotMob structure
                    team_elements = container.find_all(['span', 'div'], class_=re.compile(r'team|participant|name'))
                    if len(team_elements) >= 2:
                        home_team = self._clean_team_name(team_elements[0].get_text(strip=True))
                        away_team = self._clean_team_name(team_elements[1].get_text(strip=True))
                    
                    # Extract time
                    time_elem = container.find(['span', 'div'], class_=re.compile(r'time|date|start'))
                    match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                    
                    # Extract league/tournament
                    league_elem = container.find(['span', 'div'], class_=re.compile(r'tournament|league|competition'))
                    league = league_elem.get_text(strip=True) if league_elem else "Unknown"
                    
                    # Check if league is in our target leagues
                    if any(target_league.lower() in league.lower() for target_league in leagues):
                        fixture = MatchFixture(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=target_date.strftime('%Y-%m-%d'),
                            match_time=match_time,
                            league=league,
                            venue="TBD"
                        )
                        fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing FotMob fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing FotMob fixtures: {e}")
        
        return fixtures
    
    def _parse_espn_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse ESPN fixtures."""
        fixtures = []
        
        try:
            # Look for fixture containers
            fixture_containers = soup.find_all(['div', 'tr'], class_=re.compile(r'fixture|match|game'))
            
            for container in fixture_containers:
                try:
                    # Extract team names
                    teams = container.find_all(['span', 'a'], class_=re.compile(r'team|name'))
                    if len(teams) >= 2:
                        home_team = self._clean_team_name(teams[0].get_text(strip=True))
                        away_team = self._clean_team_name(teams[1].get_text(strip=True))
                    
                    # Extract time
                    time_elem = container.find(['span', 'div'], class_=re.compile(r'time|date'))
                    match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                    
                    # Extract league
                    league_elem = container.find(['span', 'div'], class_=re.compile(r'league|competition'))
                    league = league_elem.get_text(strip=True) if league_elem else "Unknown"
                    
                    # Check if league is in our target leagues
                    if any(target_league.lower() in league.lower() for target_league in leagues):
                        fixture = MatchFixture(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=target_date.strftime('%Y-%m-%d'),
                            match_time=match_time,
                            league=league,
                            venue="TBD"
                        )
                        fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing ESPN fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing ESPN fixtures: {e}")
        
        return fixtures
    
    def _parse_bbc_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse BBC Sport fixtures."""
        fixtures = []
        
        try:
            # BBC uses specific class names
            fixture_containers = soup.find_all('div', class_=re.compile(r'gs-o-media|gel-layout'))
            
            for container in fixture_containers:
                try:
                    # Look for team names in links or spans
                    team_links = container.find_all('a', href=re.compile(r'/sport/football/teams/'))
                    if len(team_links) >= 2:
                        home_team = self._clean_team_name(team_links[0].get_text(strip=True))
                        away_team = self._clean_team_name(team_links[1].get_text(strip=True))
                        
                        # Extract time
                        time_elem = container.find(['span', 'time'], class_=re.compile(r'time|date'))
                        match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                        
                        # Extract league from URL or text
                        league = "Premier League"  # Default for BBC
                        
                        fixture = MatchFixture(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=target_date.strftime('%Y-%m-%d'),
                            match_time=match_time,
                            league=league,
                            venue="TBD"
                        )
                        fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing BBC fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing BBC fixtures: {e}")
        
        return fixtures
    
    def _parse_sky_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse Sky Sports fixtures."""
        fixtures = []
        
        try:
            # Sky Sports uses different structure
            fixture_rows = soup.find_all('tr', class_=re.compile(r'fixture|match'))
            
            for row in fixture_rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Extract teams from first two cells
                        home_team = self._clean_team_name(cells[0].get_text(strip=True))
                        away_team = self._clean_team_name(cells[1].get_text(strip=True))
                        
                        # Extract time from third cell
                        match_time = cells[2].get_text(strip=True) if len(cells) > 2 else "TBD"
                        
                        # Extract league from context
                        league = "Premier League"  # Default for Sky Sports
                        
                        fixture = MatchFixture(
                            home_team=home_team,
                            away_team=away_team,
                            match_date=target_date.strftime('%Y-%m-%d'),
                            match_time=match_time,
                            league=league,
                            venue="TBD"
                        )
                        fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing Sky fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Sky fixtures: {e}")
        
        return fixtures
    
    def _parse_flashscore_fixtures(self, soup: BeautifulSoup, target_date: date, leagues: List[str]) -> List[MatchFixture]:
        """Parse FlashScore fixtures."""
        fixtures = []
        
        try:
            # FlashScore uses specific class names
            match_rows = soup.find_all('div', class_=re.compile(r'event__match'))
            
            for row in match_rows:
                try:
                    # Extract teams
                    home_elem = row.find('div', class_=re.compile(r'event__home'))
                    away_elem = row.find('div', class_=re.compile(r'event__away'))
                    
                    if home_elem and away_elem:
                        home_team = self._clean_team_name(home_elem.get_text(strip=True))
                        away_team = self._clean_team_name(away_elem.get_text(strip=True))
                        
                        # Extract time
                        time_elem = row.find('div', class_=re.compile(r'event__time'))
                        match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                        
                        # Extract league
                        league_elem = row.find('div', class_=re.compile(r'event__league'))
                        league = league_elem.get_text(strip=True) if league_elem else "Unknown"
                        
                        # Check if league is in our target leagues
                        if any(target_league.lower() in league.lower() for target_league in leagues):
                            fixture = MatchFixture(
                                home_team=home_team,
                                away_team=away_team,
                                match_date=target_date.strftime('%Y-%m-%d'),
                                match_time=match_time,
                                league=league,
                                venue="TBD"
                            )
                            fixtures.append(fixture)
                
                except Exception as e:
                    logger.debug(f"Error parsing FlashScore fixture: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing FlashScore fixtures: {e}")
        
        return fixtures
    
    def _clean_team_name(self, name: str) -> str:
        """Clean and standardize team names."""
        if not name:
            return "Unknown"
        
        # Remove extra whitespace and special characters
        name = re.sub(r'\s+', ' ', name.strip())
        name = re.sub(r'[^\w\s-]', '', name)
        
        # Common team name mappings
        team_mappings = {
            'manchester united': 'Manchester United',
            'man city': 'Manchester City',
            'manchester city': 'Manchester City',
            'liverpool': 'Liverpool',
            'chelsea': 'Chelsea',
            'arsenal': 'Arsenal',
            'tottenham': 'Tottenham',
            'real madrid': 'Real Madrid',
            'barcelona': 'Barcelona',
            'atletico madrid': 'Atletico Madrid',
            'bayern munich': 'Bayern Munich',
            'borussia dortmund': 'Borussia Dortmund',
            'juventus': 'Juventus',
            'ac milan': 'AC Milan',
            'inter milan': 'Inter Milan',
            'psg': 'Paris Saint-Germain',
            'paris saint germain': 'Paris Saint-Germain',
        }
        
        name_lower = name.lower()
        for key, value in team_mappings.items():
            if key in name_lower:
                return value
        
        return name.title()
    
    def _deduplicate_fixtures(self, fixtures: List[MatchFixture]) -> List[MatchFixture]:
        """Remove duplicate fixtures."""
        seen = set()
        unique_fixtures = []
        
        for fixture in fixtures:
            # Create a unique key for each fixture
            key = (fixture.home_team.lower(), fixture.away_team.lower(), fixture.match_date)
            if key not in seen:
                seen.add(key)
                unique_fixtures.append(fixture)
        
        return unique_fixtures
    
    def scrape_news_for_teams(self, teams: List[str], days_back: int = 7) -> List[NewsArticle]:
        """Scrape news articles related to specific teams."""
        all_articles = []
        
        for source_name, rss_url in self.news_sources.items():
            try:
                logger.info(f"Scraping news from {source_name}")
                articles = self._scrape_rss_news(rss_url, teams, days_back)
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to scrape news from {source_name}: {e}")
                continue
        
        # Remove duplicates and return
        unique_articles = self._deduplicate_articles(all_articles)
        logger.info(f"Found {len(unique_articles)} unique news articles")
        return unique_articles
    
    def _scrape_rss_news(self, rss_url: str, teams: List[str], days_back: int) -> List[NewsArticle]:
        """Scrape news from RSS feed."""
        articles = []
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                try:
                    # Check if article is recent enough
                    published_date = datetime(*entry.published_parsed[:6])
                    if (datetime.now() - published_date).days > days_back:
                        continue
                    
                    # Check if article is relevant to our teams
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    content = title + ' ' + summary
                    
                    relevance_score = self._calculate_relevance_score(content, teams)
                    if relevance_score < 0.3:  # Only include relevant articles
                        continue
                    
                    # Get full article content
                    full_content = self._get_article_content(entry.get('link', ''))
                    if full_content:
                        content = full_content
                    
                    # Calculate sentiment
                    sentiment_score = self._calculate_sentiment(content)
                    
                    article = NewsArticle(
                        title=title,
                        content=content,
                        url=entry.get('link', ''),
                        published_date=published_date.isoformat(),
                        source=rss_url,
                        sentiment_score=sentiment_score,
                        relevance_score=relevance_score
                    )
                    articles.append(article)
                
                except Exception as e:
                    logger.debug(f"Error processing news entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping RSS feed {rss_url}: {e}")
        
        return articles
    
    def _get_article_content(self, url: str) -> str:
        """Get full article content from URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.debug(f"Failed to get article content from {url}: {e}")
            return ""
    
    def _calculate_relevance_score(self, content: str, teams: List[str]) -> float:
        """Calculate how relevant the content is to the teams."""
        if not content or not teams:
            return 0.0
        
        content_lower = content.lower()
        relevance_score = 0.0
        
        for team in teams:
            team_lower = team.lower()
            # Check for team name mentions
            if team_lower in content_lower:
                relevance_score += 0.5
            
            # Check for player names (simplified)
            if any(player in content_lower for player in ['player', 'squad', 'team', 'match', 'game']):
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _calculate_sentiment(self, content: str) -> float:
        """Calculate sentiment score of the content."""
        if not content:
            return 0.0
        
        # Use VADER sentiment analyzer
        scores = self.sentiment_analyzer.polarity_scores(content)
        return scores['compound']  # Returns score between -1 and 1
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles."""
        seen = set()
        unique_articles = []
        
        for article in articles:
            # Create a unique key based on title and URL
            key = (article.title.lower(), article.url)
            if key not in seen:
                seen.add(key)
                unique_articles.append(article)
        
        return unique_articles
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
    
    def __del__(self):
        if self.driver:
            self.driver.quit()

class LLMNewsAnalyzer:
    """LLM-based news analysis for football predictions."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("No OpenAI API key provided. LLM analysis will be disabled.")
    
    def analyze_news_for_match(self, home_team: str, away_team: str, articles: List[NewsArticle]) -> Dict:
        """Analyze news articles for a specific match."""
        if not self.api_key or not articles:
            return self._get_default_analysis()
        
        try:
            # Prepare articles for analysis
            relevant_articles = [a for a in articles if a.relevance_score > 0.5]
            if not relevant_articles:
                return self._get_default_analysis()
            
            # Create prompt for LLM
            prompt = self._create_analysis_prompt(home_team, away_team, relevant_articles)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a football expert analyst. Analyze news articles and provide insights on team form, injuries, and match predictions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the analysis
            return self._parse_llm_analysis(analysis_text, relevant_articles)
        
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._get_default_analysis()
    
    def _create_analysis_prompt(self, home_team: str, away_team: str, articles: List[NewsArticle]) -> str:
        """Create prompt for LLM analysis."""
        prompt = f"""
        Analyze the following news articles about {home_team} vs {away_team} and provide insights:
        
        Articles:
        """
        
        for i, article in enumerate(articles[:5]):  # Limit to 5 most relevant articles
            prompt += f"\n{i+1}. {article.title}\n   Content: {article.content[:200]}...\n   Sentiment: {article.sentiment_score:.2f}\n"
        
        prompt += f"""
        
        Please provide:
        1. Team form analysis (recent performance, injuries, transfers)
        2. Key factors that could influence the match
        3. Predicted outcome with confidence level
        4. Risk factors and potential upsets
        
        Format your response as JSON with keys: form_analysis, key_factors, prediction, confidence, risks
        """
        
        return prompt
    
    def _parse_llm_analysis(self, analysis_text: str, articles: List[NewsArticle]) -> Dict:
        """Parse LLM analysis response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback to text parsing
                analysis_data = {
                    "form_analysis": analysis_text,
                    "key_factors": "Analysis available in form_analysis",
                    "prediction": "Home",
                    "confidence": 0.6,
                    "risks": "Standard match risks"
                }
            
            # Add metadata
            analysis_data.update({
                "articles_analyzed": len(articles),
                "avg_sentiment": sum(a.sentiment_score for a in articles) / len(articles),
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            return analysis_data
        
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Get default analysis when LLM is not available."""
        return {
            "form_analysis": "No recent news analysis available",
            "key_factors": "Standard match factors apply",
            "prediction": "Home",
            "confidence": 0.5,
            "risks": "Standard match risks",
            "articles_analyzed": 0,
            "avg_sentiment": 0.0,
            "analysis_timestamp": datetime.now().isoformat()
        }

# Global instances (disable Selenium by default to avoid Chrome driver dependency)
scraper = RealFootballScraper(use_selenium=False)
llm_analyzer = LLMNewsAnalyzer()
