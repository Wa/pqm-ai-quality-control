# GitHub Repository Setup Guide

This guide will help you create a private GitHub repository for the PQM_AI project.

## Prerequisites

1. **GitHub Account**: You need a GitHub account
2. **GitHub CLI**: Already installed on this system
3. **Authentication**: You need to authenticate with GitHub

## Step 1: Authenticate with GitHub

If you haven't already authenticated, run:

```bash
gh auth login
```

Follow the prompts to authenticate with your GitHub account.

## Step 2: Create the Repository

### Option A: Using the Automated Script (Recommended)

1. Choose a repository name (e.g., `pqm-ai-quality-control`)
2. Run the setup script:

```bash
./setup_github.sh pqm-ai-quality-control
```

### Option B: Manual Creation

1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `pqm-ai-quality-control` (or your preferred name)
   - **Description**: `PQM_AI Quality Control Assistant - AI-powered APQP document analysis tool`
   - **Visibility**: Private
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 3: Push Your Code

If you used the automated script, your code should already be pushed. If not, run:

```bash
# Add the remote origin (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Repository Structure

Your repository will contain:

```
PQM_AI/
├── main.py                           # Main Streamlit application
├── config.py                         # Configuration management
├── util.py                           # Utility functions
├── tab_consistency_check.py          # Consistency check tab
├── tab_file_elements_check.py        # File elements check tab
├── tab_file_completeness_check.py    # File completeness check tab
├── tab_history_issues_avoidance.py   # History issues avoidance tab
├── tab_settings.py                   # Settings tab
├── tab_help_documentation.py         # Help documentation tab
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Git ignore rules
├── demonstration/                    # Demo files
├── media/                           # Media files
└── setup_github.sh                  # GitHub setup script
```

## Security Notes

- The repository is set to **private** by default
- User-specific files (`user_settings/`, `temp.py`) are excluded via `.gitignore`
- Session directories (user uploads) are excluded via `.gitignore`
- API keys and sensitive configuration are in `config.py` (review before pushing)

## Next Steps

1. **Review Configuration**: Check `config.py` for any sensitive information
2. **Add Collaborators**: Invite team members if needed
3. **Set Up Branch Protection**: Configure branch protection rules
4. **Add Issues Template**: Create templates for bug reports and feature requests
5. **Set Up Actions**: Configure GitHub Actions for CI/CD if needed

## Troubleshooting

### Authentication Issues
```bash
gh auth logout
gh auth login
```

### Repository Already Exists
If the repository name is taken, choose a different name or add a suffix:
```bash
./setup_github.sh pqm-ai-quality-control-v1
```

### Push Issues
If you get push errors, try:
```bash
git pull origin main --allow-unrelated-histories
git push origin main
``` 