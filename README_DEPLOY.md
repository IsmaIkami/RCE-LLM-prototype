# Deployment Instructions

## Streamlit Cloud Deployment

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Configure:
   - **Repository:** `IsmaIkami/RCE-LLM-prototype`
   - **Branch:** `main`
   - **Main file:** `admin_panel.py`
4. Click "Deploy!"

## Requirements

The app will automatically install:
- streamlit
- spacy (for NLP)
- networkx (for graphs)
- numpy, plotly, python-dateutil

SpaCy model will be downloaded automatically on first run.

## Expected URL

https://rce-llm-prototype.streamlit.app
or
https://[username]-rce-llm-prototype-admin-panel-[hash].streamlit.app
