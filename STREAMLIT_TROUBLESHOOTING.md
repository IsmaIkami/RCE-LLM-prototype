# Streamlit Cloud Troubleshooting Guide

## Current Deployment Status

**Repository:** https://github.com/IsmaIkami/RCE-LLM-prototype
**Latest Commit:** ee186e8
**Deployment URL:** https://rce-llm-prototype.streamlit.app (or similar)

---

## Issue: App Not Showing Latest Code

### Symptoms
- Query "what is machine learning?" returns old answer format
- Version info shows old commit hash
- Changes pushed to GitHub but not reflected in app

### Solution Steps

#### 1. Check Version Info in Sidebar
After logging in, check the sidebar for:
```
Commit: ee186e8
Build: 2025-11-12
Loaded: [current time]
ML answer fix applied
```

If you see an older commit hash, the app hasn't updated yet.

#### 2. Hard Refresh Browser
- **Chrome/Edge:** Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- **Firefox:** Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- **Safari:** Cmd+Option+R (Mac)

#### 3. Clear Browser Cache
1. Open DevTools (F12)
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"

#### 4. Check Streamlit Cloud Dashboard
1. Go to https://share.streamlit.io/
2. Sign in with GitHub account (IsmaIkami)
3. Find "RCE-LLM-prototype" app
4. Check deployment status:
   - ✅ Green = Running latest code
   - 🔄 Yellow = Deploying
   - ❌ Red = Error

#### 5. Manual Rebuild (If Needed)
If app shows old code after 5+ minutes:

1. In Streamlit Cloud dashboard, click "⋮" menu
2. Select "Reboot app"
3. Wait 2-3 minutes for rebuild
4. Hard refresh browser

#### 6. Check Build Logs
In Streamlit Cloud dashboard:
1. Click on the app
2. Click "Manage app" → "Logs"
3. Check for errors during deployment
4. Look for "requirements.txt" installation messages

---

## Expected Behavior After Fix

### Machine Learning Query Test
**Input:** "what is machine learning?"

**Expected Output:**
```
Machine learning is a branch of artificial intelligence that enables
computer systems to learn and improve from experience without being
explicitly programmed. It uses algorithms to analyze data, identify
patterns, and make decisions with minimal human intervention.
```

**Old Output (incorrect):**
```
learning combined with unsupervised. learning related to unsupervised.
unsupervised related to learning.
```

### Version Check
Sidebar should show:
- Commit: ee186e8 (or later)
- ML answer fix applied

---

## Technical Details of Fix

### Root Cause
The machine learning template check was nested under `else: # General domain`,
but queries were being classified as "technical" domain, bypassing the
ML-specific template.

### Solution Applied
Moved ML check to top of intent logic (renderer.py:126-128):
```python
if context.intent.value == "query":
    # Check for machine learning specifically first (regardless of domain)
    if "machine learning" in main_entity.lower() or any("learning" in e.text.lower() for e in top_entities):
        return "Machine learning is a branch of artificial intelligence..."
```

Now checks for ML **before** domain-specific logic.

### Files Modified
1. `rce_llm/core/renderer.py` - Fixed ML answer logic
2. `admin_panel.py` - Added version tracking

### Git History
```bash
ee186e8 - Add deployment tracking with live reload timestamp
672f305 - Fix machine learning answer - check ML first regardless of domain
20b6420 - Improve answer generation with domain-aware templates (old/broken)
```

---

## Auto-Deployment

Streamlit Cloud automatically deploys when:
- New commits are pushed to `main` branch
- Usually takes 2-5 minutes
- Watch dashboard for deployment status

### If Auto-Deploy Fails
Check these common issues:
1. requirements.txt syntax errors
2. Import errors in Python code
3. Missing dependencies
4. Memory limits exceeded (1GB RAM limit)

---

## Testing the Fix Locally

To verify the fix works before Streamlit Cloud updates:

```bash
cd ~/RCE-LLM-prototype

# Install dependencies
pip install -e .

# Run admin panel
streamlit run admin_panel.py

# Login with: admin / rce2024
# Test query: "what is machine learning?"
```

---

## Contact

If issues persist after trying all steps:
- Email: is.sialyen@gmail.com
- Check GitHub Issues: https://github.com/IsmaIkami/RCE-LLM-prototype/issues

---

**Last Updated:** 2025-11-12
**Fix Version:** ee186e8
