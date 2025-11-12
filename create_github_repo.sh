#!/bin/bash
# Create GitHub repository using GitHub CLI

echo "Creating GitHub repository: RCE-LLM-prototype"
echo "=============================================="

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) not found. Installing..."
    brew install gh
fi

# Authenticate if needed
gh auth status || gh auth login

# Create repository
gh repo create IsmaIkami/RCE-LLM-prototype \
    --public \
    --description "Production-ready Relational Coherence Engine for Consistent and Energy-Efficient Language Modeling - DOI: 10.5281/zenodo.17360372" \
    --source=. \
    --push

echo ""
echo "✅ Repository created and pushed!"
echo "📍 URL: https://github.com/IsmaIkami/RCE-LLM-prototype"
