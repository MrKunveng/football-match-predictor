#!/bin/bash

# Football Match Predictor - GitHub Connection Script
# This script helps connect your local repository to GitHub

echo "🚀 Football Match Predictor - GitHub Setup"
echo "=========================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository. Please run this script from the football folder."
    exit 1
fi

echo "✅ Git repository detected"

# Get GitHub username
echo ""
echo "📝 Please provide your GitHub username:"
read -p "GitHub Username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Error: GitHub username is required"
    exit 1
fi

# Repository name
REPO_NAME="football-match-predictor"

echo ""
echo "🔗 Setting up remote connection to GitHub..."
echo "Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"

# Add remote origin
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

if [ $? -eq 0 ]; then
    echo "✅ Remote origin added successfully"
else
    echo "⚠️  Remote origin might already exist. Continuing..."
fi

echo ""
echo "📤 Pushing code to GitHub..."

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your code has been pushed to GitHub!"
    echo "📍 Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Sign in with your GitHub account"
    echo "3. Deploy your app using the repository: $GITHUB_USERNAME/$REPO_NAME"
    echo "4. Set main file path to: streamlit_app.py"
    echo ""
    echo "🎯 Your app will be available at: https://YOUR_APP_NAME.streamlit.app"
else
    echo ""
    echo "❌ Error pushing to GitHub. Please check:"
    echo "1. You have created the repository on GitHub"
    echo "2. Your GitHub credentials are correct"
    echo "3. You have push permissions to the repository"
    echo ""
    echo "📝 Manual steps:"
    echo "1. Create repository at: https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Make it public"
    echo "4. Don't initialize with README"
    echo "5. Run this script again"
fi
