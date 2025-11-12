#!/bin/bash
# Launch RCE-LLM Admin Panel
# Author: Ismail Sialyen

set -e

echo "========================================="
echo "  RCE-LLM Admin Panel Launcher"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install requirements if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt
pip install -q -r requirements_admin.txt
echo "✓ Dependencies installed"
echo ""

# Download spaCy model if not present
echo "Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    echo "✓ spaCy model downloaded"
}
echo ""

# Install RCE-LLM in development mode
echo "Installing RCE-LLM..."
pip install -q -e .
echo "✓ RCE-LLM installed"
echo ""

# Launch Streamlit
echo "========================================="
echo "  Launching Admin Panel..."
echo "========================================="
echo ""
echo "Admin Panel will open in your browser"
echo "Default credentials:"
echo "  Username: admin"
echo "  Password: rce2024"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run admin_panel.py --server.port 8501 --server.address localhost
