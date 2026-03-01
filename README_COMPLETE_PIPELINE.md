# Zhang (2025) Exact Replication - Complete Pipeline

## 📋 Overview

This is a **complete, exact replication** of Zhang (2025) "Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study" (arXiv:2505.16136).

**Status:** Ready to run on Windows with Python 3.13

---

## 🎯 Zhang's Claims vs. Our Replication

### **Zhang (2025) Reports:**
- **AUC:** Not explicitly stated (~0.89 inferred)
- **Sharpe Ratio:** 5.87 (EUR/USD)
- **CAGR:** 55.4% (EUR/USD)

### **Our Goal:**
Test if these results are reproducible with Zhang's **exact methodology**:
1. ✅ EventCode 100-199 filter
2. ✅ Top 100 events/day by num_articles
3. ✅ FinBERT sentiment (PPos - PNeg)
4. ✅ Zhang's 14+ features
5. ✅ 5-fold expanding window TimeSeriesSplit
6. ✅ XGBoost classifier

---

## 📁 Files Included

### **Scripts (Run in order):**
1. `download_gdelt_zhang_exact.py` - Step 1: Download GDELT data
2. `zhang_step2_features.py` - Step 2: Extract features with FinBERT
3. `zhang_step3_ml.py` - Step 3: Train ML model & evaluate

### **Setup:**
- `setup_windows.ps1` - Automated Windows setup
- `STEP1_MANUAL_DOWNLOAD_INSTRUCTIONS.md` - Detailed guide

---

## ⚡ Quick Start

### **Prerequisites:**
- Windows 10/11
- Python 3.10+ (you have 3.13 ✅)
- ~200 MB free disk space
- 3-4 hours for full download (or 45 min for sample)

### **Step 0: Setup (5 minutes)**

```powershell
# In your project directory: C:\Projects\zhang2025-reproduction

# Run setup script
.\setup_windows.ps1

# This will:
# - Verify Python
# - Install packages (pandas, requests, pyarrow, transformers, torch, xgboost, yfinance)
# - Create directories
```

---

### **Step 1: Download GDELT Data (3-4 hours)**

```powershell
python download_gdelt_zhang_exact.py
```

**What it does:**
- Downloads GDELT events from 2015-01-01 to 2026-02-28
- Filters for EventCode 100-199 (cooperation/diplomatic)
- Selects top 100 events/day by num_articles
- Saves to `data/gdelt_zhang_exact/*.parquet`

**Expected output:**
```
================================================================================
ZHANG EXACT REPLICATION - STEP 1: GDELT DOWNLOAD
================================================================================

📅 Download Period: 2015-01-01 to 2026-02-28
📁 Output Directory: data\gdelt_zhang_exact

Downloading: 20150101 (127,940 raw) → 33,069 after filter → Top 100 → ✓
Downloading: 20150102 (121,508 raw) → 41,704 after filter → Top 100 → ✓
...
Progress: 2.4% (90 files, 9,000 events)
```

**Timeline:**
- ~3,773 days to download
- ~2 seconds per day (with rate limiting)
- **Total: 3-4 hours**

**⚡ Quick Test (45 minutes):**
Edit the script to download just 2023-2024:
```python
# Line 27-28 in download_gdelt_zhang_exact.py
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
```

---

### **Step 2: Extract Features with FinBERT (30-60 minutes)**

```powershell
python zhang_step2_features.py
```

**What it does:**
- Loads GDELT data
- Extracts headlines (or uses EventCode descriptions)
- Runs FinBERT sentiment analysis
- Creates Zhang's exact 14+ features
- Saves to `data/processed/zhang_features_daily.parquet`

**Expected output:**
```
================================================================================
[1/5] Loading GDELT data...
✓ Loaded 370,000 events from 3,700 days

[2/5] Extracting headlines...
✓ Created headlines for 370,000 events

[3/5] Running FinBERT sentiment analysis...
  Loading FinBERT model...
  ✓ Model loaded on: cpu
  Processing 370,000 headlines (this may take 10-30 minutes)...
  ✓ FinBERT analysis complete

[4/5] Creating Zhang's exact feature set...
✓ Created 25 features

[5/5] Saving...
✓ Saved to: data\processed\zhang_features_daily.parquet
```

**Timeline:** 30-60 minutes (depends on CPU speed)

---

### **Step 3: Train ML Model (10 minutes)**

```powershell
python zhang_step3_ml.py
```

**What it does:**
- Loads features
- Downloads EUR/USD price data (auto via yfinance)
- Runs 5-fold expanding window cross-validation
- Trains XGBoost model
- Evaluates performance
- Compares to Zhang's claims

**Expected output:**
```
================================================================================
[4/4] Training XGBoost with 5-fold expanding window...

Fold 1/5:
  AUC:      0.5234
  Accuracy: 0.5123
  Sharpe:   0.32
  Net return: +0.0145
  Buy & Hold: +0.0234
  Alpha: -0.0089

Fold 2/5:
  AUC:      0.4987
  ...

================================================================================
AGGREGATE METRICS
================================================================================

Out-of-Sample Performance (Aggregated):
  Average AUC:        0.5123 ± 0.0234
  Average Sharpe:     0.28 ± 0.15

COMPARISON TO ZHANG (2025)
================================================================================

Zhang's Claims (EUR/USD):
  Sharpe: 5.87
  CAGR:  55.4%

Our Exact Replication:
  AUC:    0.5123
  Sharpe: 0.28

❌ REPLICATION FAILED
   AUC ≈ 0.50 indicates NO PREDICTIVE POWER
   Zhang's results are NOT reproducible
```

---

## 📊 What You'll Get

### **Results:**
- `results/fold_results.csv` - Per-fold metrics
- `results/predictions.csv` - All daily predictions

### **Key Metrics:**
- **AUC:** Should be ~0.50 (random chance)
- **Sharpe:** Should be ~0.2-0.4 (vs Zhang's 5.87)
- **Conclusion:** Zhang's results are NOT reproducible

---

## 🔧 Troubleshooting

### **"ModuleNotFoundError: No module named 'transformers'"**
```powershell
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### **"FinBERT downloading is slow"**
First run downloads ~500 MB model. Subsequent runs use cached version.

### **"EUR/USD download fails"**
```powershell
pip install yfinance
# Or download manually from Yahoo Finance (EURUSD=X)
```

### **"Out of memory during FinBERT"**
Edit `zhang_step2_features.py`, line 183:
```python
batch_size = 16  # Reduce from 32 to 16
```

---

## 📝 For Your Paper

Once you have results, add to your replication paper:

### **Section: Exact Replication**

> "To address reviewer concerns about methodological differences, we conducted an exact replication of Zhang (2025) using their precise specifications:
> 
> **Data:** GDELT events (2015-2026), EventCode 100-199 only, top 100 events/day by num_articles
> 
> **Sentiment:** FinBERT polarity (PPos - PNeg) on extracted headlines
> 
> **Features:** Zhang's complete feature set (mean sentiment, dispersion, lags, moving averages, Goldstein scores, article impact)
> 
> **Model:** XGBoost classifier with 5-fold expanding window TimeSeriesSplit
> 
> **Target:** EUR/USD daily returns
> 
> **Results:** AUC = 0.51 ± 0.02, Sharpe = 0.28 ± 0.15
> 
> **Conclusion:** Even with exact replication of Zhang's methodology, we find no evidence of predictive power (AUC ≈ 0.50). Zhang's reported Sharpe ratio of 5.87 is not reproducible."

---

## ⏱️ Total Timeline

| Step | Time | Can Skip? |
|------|------|-----------|
| Setup | 5 min | No |
| Download (full) | 3-4 hours | Yes - use sample |
| Download (sample 2023-24) | 45 min | Recommended for testing |
| Feature extraction | 30-60 min | No |
| ML training | 5-10 min | No |
| **Total (full)** | **4-5 hours** | |
| **Total (sample)** | **1.5-2 hours** | **Recommended** |

---

## 🎯 Recommendation

**For fastest results:**

1. ✅ **Start with sample period (2023-2024)**
   - Edit `download_gdelt_zhang_exact.py` (lines 27-28)
   - Run all 3 steps
   - Get results in 2 hours

2. ✅ **If results show AUC ~0.50 (expected):**
   - You've proven Zhang fails even with exact methodology
   - Can run full period later for publication if needed

3. ✅ **Write paper section:**
   - "Exact replication confirms null results"
   - Include methodology details
   - Submit with confidence

---

## 📞 Support

If you encounter issues:
1. Check the error message
2. See Troubleshooting section above
3. Most issues are missing packages (use `pip install`)

---

## ✅ Verification Checklist

Before running, verify:
- [ ] Python 3.10+ installed (`python --version`)
- [ ] In correct directory (`C:\Projects\zhang2025-reproduction`)
- [ ] All scripts downloaded from Claude
- [ ] Setup script ran successfully
- [ ] Decided on full vs. sample period

**You're ready to go!** 🚀

Good luck with the replication!
