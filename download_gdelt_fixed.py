"""
GDELT Intraday Download - FIXED VERSION
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

BASE_URL = "http://data.gdeltproject.org/gdeltv2"

# Columns to keep (by position, 0-indexed)
# 1=SQLDATE, 7=Actor1CountryCode, 26=EventCode, 34=AvgTone, 53=ActionGeo_CountryCode, 60=SOURCEURL
USECOLS = [1, 7, 26, 34, 53, 60]
COLUMN_NAMES = ["SQLDATE", "Actor1CountryCode", "EventCode", "AvgTone", "ActionGeo_CountryCode", "SOURCEURL"]

def download_file(url, output_dir):
    """Download single GDELT file with timestamp"""
    filename = url.split("/")[-1]
    timestamp_str = filename.split(".")[0]
    
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    except:
        return None, "Parse timestamp failed"
    
    # Download
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None, "404"
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, f"Download failed"
    
    # Extract
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not csv_names:
                return None, "No CSV"
            csv_data = zf.read(csv_names[0])
    except:
        return None, "Extract failed"
    
    # Parse - NO HEADER, 61 columns total
    try:
        df = pd.read_csv(
            io.BytesIO(csv_data),
            sep="\t",
            header=None,  # No header row
            usecols=USECOLS,
            dtype={1: str, 7: str, 26: float, 53: str},  # Use positions
            on_bad_lines="skip",
            encoding="latin-1",
        )
        
        # Rename columns
        df.columns = COLUMN_NAMES
        
    except Exception as e:
        return None, f"Parse: {str(e)[:50]}"
    
    if df.empty:
        return None, "Empty"
    
    # Filter for US events
    us_mask = (df["Actor1CountryCode"] == "USA") | (df["ActionGeo_CountryCode"] == "US")
    df = df[us_mask]
    
    if df.empty:
        return None, "No US events"
    
    # Add timestamp
    df['timestamp'] = timestamp
    
    # Clean EventCode
    df['EventCode'] = df['EventCode'].dropna().astype(int)
    df = df.dropna(subset=['EventCode'])
    
    if df.empty:
        return None, "No valid EventCodes"
    
    # Save
    year = timestamp.year
    month = timestamp.month
    partition_dir = output_dir / f"year={year}" / f"month={month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = partition_dir / f"gdelt_{timestamp_str}.parquet"
    df.to_parquet(out_path, index=False)
    
    return out_path, "Success"

def main():
    print("="*80)
    print("GDELT INTRADAY DOWNLOAD - FIXED")
    print("="*80)
    
    output_dir = Path('data/gdelt_intraday')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Download last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Build URLs
    urls = []
    for day in pd.date_range(start_date, end_date, freq='D'):
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                ts = f"{day.strftime('%Y%m%d')}{hour:02d}{minute:02d}00"
                urls.append(f"{BASE_URL}/{ts}.export.CSV.zip")
    
    print(f"\nDownloading {len(urls)} files...")
    print(f"Period: {start_date.date()} to {end_date.date()}\n")
    
    # Test first 5 files
    print("Testing first 5 files...")
    test_success = 0
    for url in urls[:5]:
        result, msg = download_file(url, output_dir)
        if result:
            test_success += 1
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
    
    if test_success == 0:
        print("\n❌ All test files failed!")
        return
    
    print(f"\n✓ {test_success}/5 test files successful - proceeding...\n")
    
    # Download all in parallel
    errors = {}
    success = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, url, output_dir): url for url in urls}
        
        for i, future in enumerate(as_completed(futures)):
            result, msg = future.result()
            if result:
                success += 1
            else:
                errors[msg] = errors.get(msg, 0) + 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(urls)} ({success} successful)")
    
    print(f"\n✓ Complete: {success}/{len(urls)} files downloaded")
    
    if success > 0:
        # Verify
        files = list(output_dir.rglob('*.parquet'))
        print(f"✓ {len(files)} parquet files created\n")
        
        test_file = files[0]
        df = pd.read_parquet(test_file)
        
        print("VERIFICATION:")
        print(f"  File: {test_file.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\n  Sample:")
        print(df[['timestamp', 'EventCode', 'AvgTone']].head())
        print(f"\n  EventCode distribution:")
        print(df['EventCode'].value_counts().head(10))
        
        if 'timestamp' in df.columns and 'EventCode' in df.columns:
            print("\n✅ SUCCESS! Data ready for analysis")
            print(f"\nTo copy to Docker:")
            print(f"  docker cp data/gdelt_intraday zhang2025:/app/data/")
        else:
            print("\n❌ Missing required columns")
    else:
        print("\n❌ No files downloaded")
        print("\nTop errors:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            print(f"  {error}: {count}")

if __name__ == '__main__':
    main()
