# Deploying RCE-LLM Admin Panel to Streamlit Cloud

## Quick Deployment Guide

### Step 1: Ensure Code is on GitHub ✅
Repository: https://github.com/IsmaIkami/RCE-LLM-prototype
Status: Already pushed!

### Step 2: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io/
2. Sign in with your GitHub account (IsmaIkami)

### Step 3: Deploy New App
Click **"New app"** button

### Step 4: Configure Deployment

**Repository:** `IsmaIkami/RCE-LLM-prototype`
**Branch:** `main`
**Main file path:** `admin_panel.py`

### Step 5: Advanced Settings (Optional)

**Python version:** 3.10
**Secrets:** (if needed)
```toml
# No secrets needed for now
```

### Step 6: Deploy!
Click **"Deploy!"** button

---

## Expected Deployment URL

Your app will be available at:
```
https://rce-llm-prototype.streamlit.app
```

or

```
https://ismaIkami-rce-llm-prototype-admin-panel-[hash].streamlit.app
```

---

## Files Required for Deployment ✅

All files are ready in the repository:

- ✅ `admin_panel.py` - Main Streamlit app
- ✅ `requirements.txt` - Dependencies (Streamlit Cloud compatible)
- ✅ `packages.txt` - System packages (if needed)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `rce_llm/` - Complete RCE engine package

---

## Deployment Configuration

### requirements.txt
Contains all dependencies with proper versions for Streamlit Cloud:
- spaCy with direct model URL
- All RCE-LLM dependencies
- Streamlit itself
- Pinned numpy version for compatibility

### Path Configuration
The admin panel automatically adds the correct paths:
```python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
```

---

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure `rce_llm` package structure is correct in repo

### Issue: spaCy model not loading
**Solution:** requirements.txt includes direct model URL

### Issue: Memory limits
**Solution:** Streamlit Cloud has 1GB RAM limit
- Use smaller beam size (default: 10)
- Optimize graph size limits
- Add error handling for large inputs

---

## Post-Deployment

### Custom Domain (Optional)
You can set up a custom domain in Streamlit Cloud settings

### Analytics
Enable analytics in Streamlit Cloud dashboard

### Sharing
Share the URL:
- Direct link
- Embed in documentation
- Add to paper supplementary materials

---

## Login Credentials

**Username:** admin
**Password:** rce2024

(Consider changing in production or adding proper auth)

---

## Monitoring

Check deployment logs in Streamlit Cloud dashboard:
- Build logs
- Runtime logs
- Error messages
- Resource usage

---

## Updates

To update the deployed app:
```bash
# Make changes locally
git add .
git commit -m "Update admin panel"
git push origin main
```

Streamlit Cloud will auto-redeploy!

---

## Support

If deployment issues:
1. Check Streamlit Cloud logs
2. Verify all files pushed to GitHub
3. Check requirements.txt syntax
4. Contact: is.sialyen@gmail.com

---

**Ready to deploy! Go to https://share.streamlit.io/ now!** 🚀
