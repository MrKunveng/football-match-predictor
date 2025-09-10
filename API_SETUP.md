# 🔑 API Setup Guide for Real-Time Football Data

## Overview
The app now includes real-time web scraping capabilities to fetch actual match fixtures. Here's how to set up API access for the best experience.

## 🚀 Free API Options

### 1. API-Football (Recommended)
- **Website**: https://rapidapi.com/api-sports/api/api-football
- **Free Tier**: 100 requests/day
- **Setup**:
  1. Sign up at RapidAPI
  2. Subscribe to API-Football (free tier)
  3. Get your API key
  4. Replace `YOUR_API_KEY` in `streamlit_app.py` line 246

### 2. Football-Data.org
- **Website**: https://www.football-data.org/
- **Free Tier**: 10 requests/minute
- **Setup**:
  1. Register at football-data.org
  2. Get your API token
  3. Update the API endpoint in the code

## 🔧 Configuration

### Environment Variables (Recommended)
Create a `.env` file in your project root:
```bash
# API Keys
FOOTBALL_API_KEY=your_api_key_here
FOOTBALL_DATA_TOKEN=your_token_here

# Optional: Custom API endpoints
FOOTBALL_API_URL=https://api.football-data.org/v4/matches
```

### Streamlit Secrets (For Deployment)
For Streamlit Community Cloud deployment, add to your app secrets:
```toml
# .streamlit/secrets.toml
api_keys = {
    football_api_key = "your_api_key_here"
    football_data_token = "your_token_here"
}
```

## 🌐 Web Scraping Fallbacks

The app includes multiple fallback methods:

1. **Primary**: API-Football (with your API key)
2. **Fallback 1**: ESPN fixtures scraping
3. **Fallback 2**: Football365 scraping
4. **Final Fallback**: Enhanced sample data

## 📊 Data Sources

### Current Scraping Targets:
- **ESPN**: https://www.espn.com/soccer/fixtures
- **Football365**: https://www.football365.com/fixtures
- **API-Football**: https://api.football-data.org/v4/matches

### Supported Leagues:
- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1
- Champions League

## ⚡ Performance Features

### Caching System:
- **Cache Duration**: 1 hour
- **Storage**: Session state
- **Refresh**: Manual refresh button available

### Error Handling:
- Graceful fallbacks between sources
- Timeout protection (10 seconds)
- User-friendly error messages

## 🚀 Deployment Notes

### For Streamlit Community Cloud:
1. Add API keys to secrets
2. Update code to use `st.secrets`
3. Deploy with confidence

### For Local Development:
1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys in `.env` file
3. Run: `streamlit run streamlit_app.py`

## 🔄 Manual Data Refresh

Users can manually refresh match data using the "🔄 Refresh Match Data" button in the sidebar. This will:
- Clear cached data
- Fetch fresh fixtures from web sources
- Update the display immediately

## 📝 Sample Data

When all scraping sources fail, the app falls back to enhanced sample data with realistic fixtures including:
- Barcelona vs Valencia (Sept 14, 2025) - Corrected as per user feedback
- Real Sociedad vs Real Madrid (Sept 13, 2025)
- Manchester United vs Liverpool (Sept 13, 2025)
- And many more across all leagues

## 🛠️ Troubleshooting

### Common Issues:
1. **No matches showing**: Check API key configuration
2. **Slow loading**: Enable caching or reduce scraping frequency
3. **Scraping errors**: App will automatically fall back to sample data

### Debug Mode:
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Future Enhancements

- [ ] Add more API sources
- [ ] Implement real-time score updates
- [ ] Add historical match data
- [ ] Include team statistics
- [ ] Add weather data integration

---

**🎯 Your app now fetches real-time match data with multiple fallback options for maximum reliability!**
