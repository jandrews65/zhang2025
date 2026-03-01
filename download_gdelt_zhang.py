"""
Download GDELT Data for Exact Zhang Replication
================================================

Downloads GDELT events with:
- EventCode 100-199 (conflict events)
- num_articles / num_mentions counts
- US-related events
- 2020-2023 period (matching Zhang's study period)

Run this FIRST before the replication scripts.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io
import time

print("="*80)
print("GDELT DATA DOWNLOAD FOR ZHANG REPLICATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

# Zhang's study period (adjust based on paper)
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Output directory
OUTPUT_DIR = Path('data/gdelt_daily')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"\nConfiguration:")
print(f"  Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"  Output: {OUTPUT_DIR}/")
print(f"  Filter: EventCode 100-199")

# ============================================================================
# Download Function
# ============================================================================

def download_gdelt_day(date):
    """Download GDELT export for one day"""
    
    date_str = date.strftime('%Y%m%d')
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    
    try:
        print(f"  Downloading {date_str}...", end=' ')
        
        # Download
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # GDELT V1 has 58 columns, no header
                # Columns we need:
                # 0: GLOBALEVENTID
                # 1: SQLDATE
                # 26: EventCode
                # 27: EventBaseCode
                # 31: NumMentions
                # 32: NumSources
                # 33: NumArticles
                # 34: AvgTone
                # 53: SOURCEURL
                
                # Read specific columns only
                df = pd.read_csv(f, sep='\t', header=None, 
                               usecols=[0, 1, 5, 6, 16, 17, 26, 31, 32, 33, 34, 53],
                               names=['GLOBALEVENTID', 'SQLDATE', 
                                     'Actor1Code', 'Actor1Name',
                                     'Actor2Code', 'Actor2Name',
                                     'EventCode', 
                                     'NumMentions', 'NumSources', 'NumArticles', 
                                     'AvgTone', 'SOURCEURL'],
                               dtype={'EventCode': str})
        
        # Filter EventCode 100-199
        df['EventCode'] = pd.to_numeric(df['EventCode'], errors='coerce')
        df = df[df['EventCode'].between(100, 199)]
        
        # Filter US-related (either actor is US)
        us_mask = (
            (df['Actor1Code'].str.contains('USA', na=False)) |
            (df['Actor2Code'].str.contains('USA', na=False))
        )
        df = df[us_mask]
        
        # Save
        if len(df) > 0:
            output_file = OUTPUT_DIR / f'gdelt_{date_str}.parquet'
            df.to_parquet(output_file)
            print(f"✓ {len(df)} events")
            return True
        else:
            print("✗ No events after filtering")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Parse error: {e}")
        return False

# ============================================================================
# Main Download Loop
# ============================================================================

print(f"\nStarting download...")
print(f"This will take ~{(END_DATE - START_DATE).days} requests")
print("")

current_date = START_DATE
success_count = 0
fail_count = 0

while current_date <= END_DATE:
    # Check if already downloaded
    output_file = OUTPUT_DIR / f'gdelt_{current_date.strftime("%Y%m%d")}.parquet'
    
    if output_file.exists():
        print(f"  {current_date.strftime('%Y%m%d')} already exists, skipping")
        success_count += 1
    else:
        # Download
        if download_gdelt_day(current_date):
            success_count += 1
        else:
            fail_count += 1
        
        # Rate limiting (be nice to GDELT servers)
        time.sleep(1)
    
    current_date += timedelta(days=1)
    
    # Progress update every 30 days
    if (current_date - START_DATE).days % 30 == 0:
        print(f"\n  Progress: {current_date.date()} ({success_count} success, {fail_count} failed)\n")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)

print(f"\nResults:")
print(f"  Successful: {success_count}")
print(f"  Failed: {fail_count}")
print(f"  Total files: {len(list(OUTPUT_DIR.glob('*.parquet')))}")

# Load one file to check
sample_file = sorted(OUTPUT_DIR.glob('*.parquet'))[0]
sample = pd.read_parquet(sample_file)

print(f"\nSample data ({sample_file.name}):")
print(f"  Events: {len(sample)}")
print(f"  Columns: {list(sample.columns)}")
print(f"\nEventCode distribution:")
print(sample['EventCode'].value_counts().head(10))

print(f"\n✓ Data ready for replication")
print(f"   Next: Run zhang_exact_replication.py")
