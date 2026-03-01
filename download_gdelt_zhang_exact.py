#!/usr/bin/env python3
"""
GDELT Download Script - Zhang (2025) Exact Replication
========================================================

Downloads GDELT data with Zhang's exact specifications:
- EventCode 100-199 (cooperation/diplomatic events)
- Period: 2015-01-01 to 2025-04-30 (Zhang's study period)
- Filters for num_articles/num_mentions
- Extracts URLs for headline scraping

This is Step 1 of the exact replication.

Author: Exact Zhang Replication
Date: 2026-02-28
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io
import time
import sys

print("="*80)
print("ZHANG EXACT REPLICATION - STEP 1: GDELT DOWNLOAD")
print("="*80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration (Zhang's exact parameters)
# ============================================================================

# Zhang's study period: "January 1, 2015, to April 30, 2025"
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2025, 4, 30)

# But let's be realistic - we can't get future data!
# Adjust END_DATE to today or a reasonable past date
TODAY = datetime.now()
if END_DATE > TODAY:
    print(f"\n⚠ Zhang's end date (2025-04-30) is in the future!")
    print(f"  Adjusting to most recent complete day: {TODAY.strftime('%Y-%m-%d')}")
    END_DATE = TODAY - timedelta(days=1)  # Yesterday (most recent complete data)

# Output directory
OUTPUT_DIR = Path('data/gdelt_zhang_exact')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n📅 Download Period:")
print(f"   Start: {START_DATE.strftime('%Y-%m-%d')}")
print(f"   End:   {END_DATE.strftime('%Y-%m-%d')}")
print(f"   Days:  {(END_DATE - START_DATE).days + 1}")

print(f"\n📁 Output Directory: {OUTPUT_DIR.absolute()}")

print(f"\n🔍 Zhang's Filters:")
print(f"   - EventCode: 100-199 (cooperation/diplomatic)")
print(f"   - Top 100 events/day by num_articles")
print(f"   - Extract headlines from URLs")

# ============================================================================
# GDELT Column Names (V1 Export format)
# ============================================================================

# GDELT V1 has 58 columns, no header
# We only need specific columns
GDELT_COLUMNS = {
    0: 'GLOBALEVENTID',
    1: 'SQLDATE',
    5: 'Actor1Code',
    6: 'Actor1Name', 
    15: 'Actor2Code',
    16: 'Actor2Name',
    26: 'EventCode',
    27: 'EventBaseCode',
    30: 'GoldsteinScale',
    31: 'NumMentions',
    32: 'NumSources',
    33: 'NumArticles',
    34: 'AvgTone',
    53: 'SOURCEURL'
}

# ============================================================================
# Download Function
# ============================================================================

def download_gdelt_day(date):
    """
    Download and filter GDELT data for one day
    
    Returns:
        DataFrame with filtered events, or None if error/no data
    """
    
    date_str = date.strftime('%Y%m%d')
    
    # GDELT V1 URL format
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    
    try:
        print(f"  {date_str}: Downloading...", end=' ', flush=True)
        
        # Download with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            
            # Read only the columns we need
            df = pd.read_csv(
                z.open(csv_name),
                sep='\t',
                header=None,
                usecols=list(GDELT_COLUMNS.keys()),
                names=list(GDELT_COLUMNS.values()),
                dtype={'EventCode': str, 'EventBaseCode': str},
                low_memory=False
            )
        
        print(f"({len(df):,} raw events) →", end=' ', flush=True)
        
        # ================================================================
        # FILTER 1: EventCode 100-199 (Zhang's exact filter)
        # ================================================================
        
        df['EventCode'] = pd.to_numeric(df['EventCode'], errors='coerce')
        df = df[df['EventCode'].between(100, 199)]
        
        print(f"{len(df):,} after EventCode filter →", end=' ', flush=True)
        
        if len(df) == 0:
            print("✗ No events after filtering")
            return None
        
        # ================================================================
        # FILTER 2: Ensure we have article counts
        # ================================================================
        
        # Zhang uses num_articles to rank events
        # If NumArticles is missing, use NumMentions as proxy
        if df['NumArticles'].isna().all():
            print("using NumMentions as proxy →", end=' ', flush=True)
            df['num_articles'] = df['NumMentions'].fillna(0)
        else:
            df['num_articles'] = df['NumArticles'].fillna(df['NumMentions'].fillna(0))
        
        # ================================================================
        # FILTER 3: Top 100 events by num_articles (Zhang's exact filter)
        # ================================================================
        
        # Sort by num_articles descending
        df = df.sort_values('num_articles', ascending=False)
        
        # Take top 100 (or fewer if less than 100 events that day)
        df = df.head(100).copy()
        
        print(f"Top {len(df)} events →", end=' ', flush=True)
        
        # ================================================================
        # Clean and prepare
        # ================================================================
        
        # Add date column
        df['date'] = date
        
        # Ensure SOURCEURL exists (needed for headline extraction)
        if 'SOURCEURL' not in df.columns or df['SOURCEURL'].isna().all():
            print("⚠ No URLs available")
            df['SOURCEURL'] = ''
        
        # Select final columns
        df = df[[
            'date', 'GLOBALEVENTID', 'EventCode', 'GoldsteinScale',
            'Actor1Code', 'Actor1Name', 'Actor2Code', 'Actor2Name',
            'num_articles', 'NumMentions', 'NumSources', 'AvgTone',
            'SOURCEURL'
        ]]
        
        print(f"✓ Saved")
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"✗ No data (404)")
        else:
            print(f"✗ HTTP {e.response.status_code}")
        return None
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return None

# ============================================================================
# Main Download Loop
# ============================================================================

print(f"\n{'='*80}")
print("STARTING DOWNLOAD")
print(f"{'='*80}\n")

current_date = START_DATE
success_count = 0
fail_count = 0
total_events = 0

all_data = []

while current_date <= END_DATE:
    # Check if already downloaded
    date_str = current_date.strftime('%Y%m%d')
    output_file = OUTPUT_DIR / f'gdelt_{date_str}.parquet'
    
    if output_file.exists():
        print(f"  {date_str}: Already exists, skipping")
        try:
            existing = pd.read_parquet(output_file)
            success_count += 1
            total_events += len(existing)
        except:
            pass
    else:
        # Download
        df = download_gdelt_day(current_date)
        
        if df is not None and len(df) > 0:
            # Save to parquet
            df.to_parquet(output_file, index=False)
            success_count += 1
            total_events += len(df)
            all_data.append(df)
        else:
            fail_count += 1
        
        # Rate limiting (be nice to GDELT servers)
        time.sleep(2)
    
    current_date += timedelta(days=1)
    
    # Progress update every 30 days
    days_done = (current_date - START_DATE).days
    if days_done % 30 == 0:
        pct = (days_done / (END_DATE - START_DATE).days) * 100
        print(f"\n  📊 Progress: {pct:.1f}% ({success_count} files, {total_events:,} events)\n")

# ============================================================================
# Summary and Validation
# ============================================================================

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)

print(f"\n📊 Results:")
print(f"   Successful:     {success_count:,} days")
print(f"   Failed:         {fail_count:,} days")
print(f"   Total events:   {total_events:,}")
print(f"   Avg events/day: {total_events/max(success_count,1):.1f}")

# List downloaded files
parquet_files = sorted(OUTPUT_DIR.glob('*.parquet'))
print(f"\n📁 Files created: {len(parquet_files)}")

if len(parquet_files) > 0:
    # Load one sample to show structure
    sample_file = parquet_files[len(parquet_files)//2]  # Middle file
    sample = pd.read_parquet(sample_file)
    
    print(f"\n📄 Sample file: {sample_file.name}")
    print(f"   Events: {len(sample)}")
    print(f"   Date: {sample['date'].iloc[0]}")
    print(f"\n   Columns:")
    for col in sample.columns:
        print(f"     - {col}")
    
    print(f"\n   EventCode distribution (sample day):")
    event_counts = sample['EventCode'].value_counts().head(5)
    for code, count in event_counts.items():
        print(f"     {int(code):3d}: {count:2d} events")
    
    print(f"\n   Article counts (sample day):")
    print(f"     Max:    {sample['num_articles'].max():.0f}")
    print(f"     Median: {sample['num_articles'].median():.0f}")
    print(f"     Min:    {sample['num_articles'].min():.0f}")
    
    # Check URL availability
    has_url = sample['SOURCEURL'].notna() & (sample['SOURCEURL'] != '')
    print(f"\n   URLs available: {has_url.sum()}/{len(sample)} ({has_url.mean()*100:.1f}%)")

print(f"\n✅ Data ready for Step 2: Feature extraction")
print(f"   Next script: zhang_exact_replication.py")
print(f"\n   Output directory: {OUTPUT_DIR.absolute()}")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
