# 🚀 GitHub Repository Setup Instructions

## Step 1: Create GitHub Repository

1. **Go to GitHub**: Visit [github.com](https://github.com) and sign in
2. **Create New Repository**: Click the "+" icon → "New repository"
3. **Repository Settings**:
   - **Repository name**: `football-match-predictor`
   - **Description**: `⚽ Interactive Football Match Predictor with AI-powered predictions`
   - **Visibility**: Public (for Streamlit Community Cloud)
   - **Initialize**: ❌ Don't initialize with README, .gitignore, or license (we already have these)

4. **Click "Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/football-match-predictor.git

# Push your code to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. **Refresh your GitHub repository page**
2. **Verify all files are uploaded**:
   - ✅ `streamlit_app.py`
   - ✅ `requirements.txt`
   - ✅ `.streamlit/config.toml`
   - ✅ `README.md`
   - ✅ `.gitignore`

## Step 4: Deploy to Streamlit Community Cloud

1. **Go to Streamlit**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. **Deploy New App**:
   - Click "New app"
   - **Repository**: Select `YOUR_USERNAME/football-match-predictor`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose your custom URL (e.g., `football-predictor`)

4. **Click "Deploy"**

## Step 5: Test Your Deployment

1. **Wait 2-5 minutes** for deployment to complete
2. **Visit your app**: `https://YOUR_APP_NAME.streamlit.app`
3. **Test the features**:
   - Select teams from different leagues
   - Set match date and time
   - Generate predictions
   - View probability charts

## 🎯 Your App Will Be Live At:
`https://YOUR_APP_NAME.streamlit.app`

## 📝 Repository Structure:
```
football-match-predictor/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── README.md                # Project documentation
├── .gitignore              # Git ignore rules
└── setup_github.md         # This file
```

## 🔧 Future Updates:

To update your deployed app:
```bash
# Make changes to your code
git add .
git commit -m "Update: Description of changes"
git push origin main
```

Streamlit Community Cloud will automatically redeploy your app!

## 🆘 Troubleshooting:

- **Deployment fails**: Check the logs in Streamlit Community Cloud
- **App not loading**: Verify all files are in the repository
- **Import errors**: Check `requirements.txt` has all dependencies
- **Styling issues**: Verify `.streamlit/config.toml` is present

---

**🎉 Congratulations! Your Football Match Predictor is now live on the web! ⚽**
