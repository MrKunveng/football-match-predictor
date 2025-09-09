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

# Popular teams for dropdown
POPULAR_TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds United",
        "Leicester City", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur",
        "West Ham United", "Wolverhampton Wanderers"
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Sociedad",
        "Real Betis", "Villarreal", "Valencia", "Athletic Bilbao", "Osasuna",
        "Rayo Vallecano", "Celta Vigo", "Mallorca", "Girona", "Getafe",
        "Espanyol", "Cadiz", "Almeria", "Valladolid", "Elche"
    ],
    "Serie A": [
        "Juventus", "AC Milan", "Inter Milan", "Napoli", "Atalanta",
        "Roma", "Lazio", "Fiorentina", "Torino", "Bologna",
        "Udinese", "Sassuolo", "Empoli", "Monza", "Lecce",
        "Salernitana", "Spezia", "Verona", "Cremonese", "Sampdoria"
    ],
    "Bundesliga": [
        "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Union Berlin",
        "Freiburg", "Bayer Leverkusen", "Eintracht Frankfurt", "Wolfsburg",
        "Mainz 05", "Borussia Mönchengladbach", "Cologne", "Werder Bremen",
        "Bochum", "Augsburg", "VfB Stuttgart", "Hertha Berlin", "Schalke 04"
    ],
    "Ligue 1": [
        "Paris Saint-Germain", "Marseille", "Monaco", "Lens", "Rennes",
        "Lille", "Lorient", "Clermont", "Lyon", "Toulouse",
        "Reims", "Montpellier", "Troyes", "Brest", "Strasbourg",
        "Nantes", "Auxerre", "Ajaccio", "Angers", "Nice"
    ]
}

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

def get_prediction_from_api(home_team: str, away_team: str, match_date: date, match_time: time) -> Optional[Dict]:
    """Get prediction from the FastAPI backend."""
    try:
        # Combine date and time
        match_datetime = datetime.combine(match_date, match_time)
        
        # Prepare request data
        request_data = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date.isoformat(),
            "league": "premier_league",  # Default league
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
            return get_mock_prediction(home_team, away_team, match_datetime)
            
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

def get_mock_prediction(home_team: str, away_team: str, match_datetime: datetime) -> Dict:
    """Generate a mock prediction when API is not available."""
    # Simple mock prediction based on team names
    np.random.seed(hash(home_team + away_team) % 2**32)
    
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
        "match_date": match_datetime.date().isoformat(),
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

def display_prediction_result(prediction: Dict):
    """Display prediction results in a nice format."""
    if not prediction:
        return
    
    # Main prediction card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    # Match info
    st.markdown(f"""
    <div class="match-info">
        <div class="team-name">{prediction['home_team']} vs {prediction['away_team']}</div>
        <p><strong>Match Date:</strong> {prediction['match_date']}</p>
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
    
    st.plotly_chart(fig, use_container_width=True)
    
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
        with st.expander(f"Match {len(st.session_state.predictions_history) - i}: {pred['home_team']} vs {pred['away_team']}"):
            display_prediction_result(pred)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Predict football match outcomes using advanced machine learning models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.header("🎯 Match Details")
        
        # League selection
        league = st.selectbox("Select League", list(POPULAR_TEAMS.keys()))
        
        # Team selection
        home_team = st.selectbox("Home Team", POPULAR_TEAMS[league], key="home_team")
        away_team = st.selectbox("Away Team", [team for team in POPULAR_TEAMS[league] if team != home_team], key="away_team")
        
        # Date and time selection
        st.subheader("📅 Match Schedule")
        
        # Date selection
        match_date = st.date_input(
            "Match Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365)
        )
        
        # Time selection
        match_time = st.time_input(
            "Match Time",
            value=time(15, 0),  # Default 3:00 PM
            step=timedelta(minutes=30)
        )
        
        # Additional options
        st.subheader("⚙️ Options")
        show_confidence = st.checkbox("Show Confidence Analysis", value=True)
        save_prediction = st.checkbox("Save to History", value=True)
        
        # Predict button
        predict_button = st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True)
    
    # Main content area
    if predict_button:
        if home_team == away_team:
            st.error("Please select different teams for home and away!")
        else:
            # Show loading spinner
            with st.spinner("Analyzing match and generating prediction..."):
                # Get prediction
                prediction = get_prediction_from_api(home_team, away_team, match_date, match_time)
                
                if prediction:
                    # Display prediction
                    st.success("✅ Prediction generated successfully!")
                    display_prediction_result(prediction)
                    
                    # Save to history if requested
                    if save_prediction:
                        st.session_state.predictions_history.append(prediction)
                        st.success("💾 Prediction saved to history!")
                    
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
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
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
                        st.plotly_chart(fig_pie, use_container_width=True)
    
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
