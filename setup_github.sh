#!/bin/bash

# Script to set up GitHub repository for PQM_AI project
# Run this script after creating a private repository on GitHub

echo "🚀 Setting up GitHub repository for PQM_AI project"
echo ""

# Check if repository name is provided
if [ -z "$1" ]; then
    echo "❌ Please provide the repository name as an argument"
    echo "Usage: ./setup_github.sh <repository-name>"
    echo "Example: ./setup_github.sh pqm-ai-quality-control"
    exit 1
fi

REPO_NAME=$1
GITHUB_USERNAME=$(gh api user --jq .login 2>/dev/null)

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Not authenticated with GitHub. Please run 'gh auth login' first."
    exit 1
fi

echo "✅ Authenticated as: $GITHUB_USERNAME"
echo "📦 Repository name: $REPO_NAME"
echo ""

# Create the repository on GitHub
echo "🔧 Creating private repository on GitHub..."
gh repo create "$REPO_NAME" --private --description "PQM_AI Quality Control Assistant - AI-powered APQP document analysis tool" --source=. --remote=origin --push

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Repository created successfully!"
    echo "🌐 Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "📋 Next steps:"
    echo "1. Visit the repository URL to verify it was created correctly"
    echo "2. Add collaborators if needed"
    echo "3. Set up branch protection rules if desired"
    echo ""
    echo "🎉 Your PQM_AI project is now on GitHub!"
else
    echo ""
    echo "❌ Failed to create repository. Please check:"
    echo "1. You have the necessary permissions on GitHub"
    echo "2. The repository name is available"
    echo "3. You're properly authenticated"
fi 