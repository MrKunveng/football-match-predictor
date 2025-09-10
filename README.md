# ⚽ Football Match Predictor - Streamlit App

A beautiful, interactive web application for predicting football match outcomes using machine learning models.

## 🚀 Features

- **Scheduled Match Display**: View real football fixtures with predetermined times and venues
- **Multiple Leagues**: Support for Premier League, La Liga, Serie A, Bundesliga, Ligue 1, and Champions League
- **Date-Based Browsing**: Select any date to see scheduled matches for that day
- **League Filtering**: Filter matches by specific leagues
- **Individual Predictions**: Get AI predictions for each scheduled match
- **Quick Previews**: Instant prediction previews without full analysis
- **Visual Predictions**: Beautiful charts showing probability distributions
- **Confidence Analysis**: Gauge charts and confidence indicators
- **Prediction History**: Save and view your prediction history
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 How to Use

1. **Select Date**: Choose a date from the sidebar to view scheduled matches
2. **Filter Leagues**: Use the league filter to show only specific competitions
3. **View Matches**: See all scheduled matches with times, venues, and teams
4. **Get Predictions**: Click "🔮 Predict" for detailed AI analysis or "👁️ Preview" for quick results
5. **View Results**: See probabilities, confidence levels, and visualizations
6. **Save History**: Your predictions are automatically saved for future reference

## 🛠️ Deployment on Streamlit Community Cloud

### Prerequisites
- GitHub repository with your code
- Streamlit Community Cloud account

### Deployment Steps

1. **Prepare Your Repository**:
   ```bash
   # Ensure your main app file is named streamlit_app.py
   # Include requirements-streamlit.txt
   # Add .streamlit/config.toml for configuration
   ```

2. **Deploy to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the repository and branch
   - Set main file path to `streamlit_app.py`
   - Click "Deploy"

3. **Configuration**:
   - The app will automatically use `requirements-streamlit.txt`
   - Custom theme is applied via `.streamlit/config.toml`
   - Mock predictions work without external API

## 📁 File Structure

```
├── streamlit_app.py              # Main Streamlit application
├── requirements-streamlit.txt    # Minimal dependencies for deployment
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── README-streamlit.md          # This file
└── src/                         # Source code (optional for deployment)
    ├── models/                  # ML models
    ├── api/                     # FastAPI backend
    └── features/                # Feature engineering
```

## 🔧 Configuration

### Environment Variables
The app can be configured with environment variables:

```bash
# Optional: API URL for backend predictions
STREAMLIT_API_URL=https://your-api-url.com

# Optional: Default league
STREAMLIT_DEFAULT_LEAGUE=Premier League
```

### Customization
- **Teams**: Modify `POPULAR_TEAMS` dictionary in `streamlit_app.py`
- **Styling**: Update CSS in the `st.markdown()` sections
- **Theme**: Modify `.streamlit/config.toml`

## 🎨 Features Overview

### 1. Team Selection
- Dropdown menus with popular teams from major leagues
- Prevents selecting the same team for home and away

### 2. Date & Time Selection
- Date picker with validation (future dates only)
- Time picker with 30-minute intervals
- Default to tomorrow at 3:00 PM

### 3. Prediction Display
- **Probability Bars**: Visual representation of win/draw/loss probabilities
- **Confidence Gauge**: Interactive gauge showing prediction confidence
- **Pie Chart**: Probability distribution visualization
- **Match Info Card**: Clean display of match details and prediction

### 4. Prediction History
- Saves up to 5 recent predictions
- Expandable cards showing full prediction details
- Persistent across app sessions

## 🔮 Prediction Models

The app supports multiple prediction modes:

1. **API Mode**: Connects to FastAPI backend for ML predictions
2. **Mock Mode**: Generates realistic mock predictions when API unavailable
3. **Fallback**: Always provides predictions even without backend

## 🎯 Use Cases

- **Football Fans**: Get predictions for upcoming matches
- **Betting Analysis**: View probability distributions
- **Match Planning**: Plan viewing schedules with predictions
- **Data Analysis**: Explore prediction patterns and confidence

## 🚀 Performance

- **Fast Loading**: Minimal dependencies for quick deployment
- **Responsive**: Works on all device sizes
- **Caching**: Efficient data handling and caching
- **Error Handling**: Graceful fallbacks and error messages

## 🔒 Security

- **Input Validation**: All user inputs are validated
- **CORS Protection**: Secure API communication
- **Error Handling**: No sensitive information in error messages

## 📊 Analytics

The app tracks:
- Prediction requests
- User interactions
- Error rates
- Performance metrics

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements-streamlit.txt

# Run locally
streamlit run streamlit_app.py

# Run with custom port
streamlit run streamlit_app.py --server.port 8502
```

### Testing
```bash
# Test the app
streamlit run streamlit_app.py

# Check different leagues and teams
# Verify date/time selection
# Test prediction generation
```

## 📈 Future Enhancements

- [ ] Real-time match updates
- [ ] Live betting odds integration
- [ ] Team form analysis
- [ ] Head-to-head statistics
- [ ] Weather impact analysis
- [ ] Social media sentiment
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
- Check the GitHub issues
- Review the Streamlit documentation
- Contact the development team

---

**⚽ Enjoy predicting football matches with AI! ⚽**
