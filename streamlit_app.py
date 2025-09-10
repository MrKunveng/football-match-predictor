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

def get_mock_prediction(home_team: str, away_team: str, match_date: str, match_time: str, league: str) -> Dict:
    """Generate a mock prediction when API is not available."""
    # Simple mock prediction based on team names
    np.random.seed(hash(home_team + away_team + league) % 2**32)
    
    # Generate realistic probabilities
    home_prob = np.random.beta(2, 3)  # Slightly favor home team
    draw_prob = np.random.beta(1, 4)  # Lower draw probability
    away_prob = 1 - home_prob - draw_prob
    
    # Ensure probabilities sum to 1
    total = home_prob + draw_prob + away_prob
    home_prob /= total
    draw_prob /= total
    away_prob /= total
    
    # Determine predicted outcome
    probs = [home_prob, draw_prob, away_prob]
    predicted_outcome = ['Home', 'Draw', 'Away'][np.argmax(probs)]
    confidence = max(probs)
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "match_time": match_time,
        "league": league,
        "predictions": {
            "home_win": round(home_prob, 3),
            "draw": round(draw_prob, 3),
            "away_win": round(away_prob, 3)
        },
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 3),
        "model_used": "Mock Model (API not available)",
        "timestamp": datetime.now().isoformat()
    }

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
        
        # Additional options
        st.subheader("⚙️ Options")
        show_confidence = st.checkbox("Show Confidence Analysis", value=True)
        save_prediction = st.checkbox("Save to History", value=True)
    
    # Main content area
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Get matches for selected date
    if date_str in SAMPLE_FIXTURES:
        matches = SAMPLE_FIXTURES[date_str]
        
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
                                # Generate quick preview
                                preview_pred = get_mock_prediction(
                                    match['home_team'], 
                                    match['away_team'], 
                                    date_str, 
                                    match['match_time'], 
                                    match['league']
                                )
                                preview_pred['venue'] = match['venue']
                                
                                # Show quick preview
                                st.info(f"Quick Preview: {preview_pred['predicted_outcome']} ({preview_pred['confidence']:.1%})")
                        
                        st.markdown("---")
        else:
            st.info(f"No matches found for {selected_date.strftime('%B %d, %Y')} in the selected leagues.")
    else:
        st.info(f"No matches scheduled for {selected_date.strftime('%B %d, %Y')}.")
    
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
