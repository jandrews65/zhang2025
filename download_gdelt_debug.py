"""
GDELT Intraday Download - Debug Version
"""

import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import requests

BASE_URL = "http://data.gdeltproject.org/gdeltv2"

GDELT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

USECOLS = [1, 7, 26, 34, 53, 60]

def download_file(url, output_dir):
    """Download single GDELT file with timestamp"""
    filename = url.split("/")[-1]
    timestamp_str = filename.split(".")[0]
    
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    except Exception as e:
        return None, f"Parse timestamp failed: {e}"
    
    # Download
    try:
        resp = requests.get(url, timeout=60)  # Increased timeout
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return None, "Timeout"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None, "404 Not Found"
        return None, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return None, f"Download failed: {e}"
    
    # Extract
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not csv_names:
                return None, "No CSV in zip"
            csv_data = zf.read(csv_names[0])
    except Exception as e:
        return None, f"Extract failed: {e}"
    
    # Parse
    try:
        df = pd.read_csv(
            io.BytesIO(csv_data),
            sep="\t",
            header=None,
            names=GDELT_COLUMNS,
            usecols=USECOLS,
            dtype={"SQLDATE": str, "Actor1CountryCode": str, "EventCode": float},  # float to handle NaN
            on_bad_lines="skip",
            encoding="latin-1",
        )
    except Exception as e:
        return None, f"Parse failed: {e}"
    
    if df.empty:
        return None, "Empty after parse"
    
    # Filter
    us_mask = (df.iloc[:, 1] == "USA") | (df.iloc[:, 4] == "US")
    df = df[us_mask]
    
    if df.empty:
        return None, "Empty after filter"
    
    df.columns = ["SQLDATE", "Actor1CountryCode", "EventCode", "AvgTone", "ActionGeo_CountryCode", "SOURCEURL"]
    df['timestamp'] = timestamp
    
    # Convert EventCode to int, drop NaN
    df['EventCode'] = df['EventCode'].dropna().astype(int)
    
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
    print("GDELT INTRADAY DOWNLOAD - DEBUG VERSION")
    print("="*80)
    
    output_dir = Path('data/gdelt_intraday')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Try just last 2 days first
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    # Build URLs
    urls = []
    for day in pd.date_range(start_date, end_date, freq='D'):
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                ts = f"{day.strftime('%Y%m%d')}{hour:02d}{minute:02d}00"
                urls.append(f"{BASE_URL}/{ts}.export.CSV.zip")
    
    print(f"\nTesting with {len(urls)} files (last 2 days)...")
    print(f"Period: {start_date.date()} to {end_date.date()}\n")
    
    # Track error types
    errors = {}
    success = 0
    
    # Try first 10 files sequentially for debugging
    print("Testing first 10 files sequentially...")
    for i, url in enumerate(urls[:10]):
        result, msg = download_file(url, output_dir)
        if result:
            success += 1
            print(f"  ✓ {i+1}: Success")
        else:
            errors[msg] = errors.get(msg, 0) + 1
            print(f"  ✗ {i+1}: {msg}")
    
    if success == 0:
        print("\n❌ All test files failed!")
        print("\nError summary:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
        
        print("\n💡 Possible solutions:")
        print("  1. Check internet connection")
        print("  2. Try using a VPN")
        print("  3. GDELT server might be down (check: http://data.gdeltproject.org/gdeltv2/)")
        return
    
    print(f"\n✓ {success}/10 test files successful!")
    print("Proceeding with full download...\n")
    
    # Download all in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers
        futures = {executor.submit(download_file, url, output_dir): url for url in urls}
        
        for i, future in enumerate(as_completed(futures)):
            result, msg = future.result()
            if result:
                success += 1
            else:
                errors[msg] = errors.get(msg, 0) + 1
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(urls)} ({success} successful)")
    
    print(f"\n✓ Complete: {success}/{len(urls)} files downloaded")
    
    if success == 0:
        print("\n❌ No files downloaded!")
        print("\nError summary:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
        return
    
    # Verify
    files = list(output_dir.rglob('*.parquet'))
    print(f"\n✓ {len(files)} parquet files created")
    
    test_file = files[0]
    df = pd.read_parquet(test_file)
    
    print("\nVERIFICATION:")
    print(f"  File: {test_file.name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  Sample:")
    print(df[['timestamp', 'EventCode', 'AvgTone']].head())
    
    if 'timestamp' in df.columns and 'EventCode' in df.columns:
        print("\n✅ SUCCESS! Ready for analysis")
        print(f"\nNext step: Copy to Docker")
        print(f"  docker cp {output_dir} zhang2025:/app/data/")
    else:
        print("\n❌ FAILED - Missing required columns")

if __name__ == '__main__':
    main()
