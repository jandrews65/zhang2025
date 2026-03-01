"""
GDELT v2 ingestion WITH timestamps - Windows version
Run this OUTSIDE Docker on your Windows machine
"""

import io
import json
import time
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
    except:
        return None
    
    # Download
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except:
        return None
    
    # Extract
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not csv_names:
                return None
            csv_data = zf.read(csv_names[0])
    except:
        return None
    
    # Parse
    try:
        df = pd.read_csv(
            io.BytesIO(csv_data),
            sep="\t",
            header=None,
            names=GDELT_COLUMNS,
            usecols=USECOLS,
            dtype={"SQLDATE": str, "Actor1CountryCode": str, "EventCode": int},
            on_bad_lines="skip",
            encoding="latin-1",
        )
    except:
        return None
    
    if df.empty:
        return None
    
    # Filter
    us_mask = (df.iloc[:, 1] == "USA") | (df.iloc[:, 4] == "US")
    df = df[us_mask]
    
    if df.empty:
        return None
    
    df.columns = ["SQLDATE", "Actor1CountryCode", "EventCode", "AvgTone", "ActionGeo_CountryCode", "SOURCEURL"]
    df['timestamp'] = timestamp
    
    # Save
    year = timestamp.year
    month = timestamp.month
    partition_dir = output_dir / f"year={year}" / f"month={month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = partition_dir / f"gdelt_{timestamp_str}.parquet"
    df.to_parquet(out_path, index=False)
    
    return out_path

def main():
    print("="*80)
    print("GDELT INTRADAY DOWNLOAD (Last 7 days)")
    print("="*80)
    
    output_dir = Path('data/gdelt_intraday')
    output_dir.mkdir(exist_ok=True)
    
    # Last 7 days
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
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    success = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, url, output_dir): url for url in urls}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                success += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(urls)} ({success} successful)")
    
    print(f"\n✓ Complete: {success} files downloaded")
    
    # Verify
    test_file = list(output_dir.rglob('*.parquet'))[0]
    df = pd.read_parquet(test_file)
    
    print("\nVERIFICATION:")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample:\n{df[['timestamp', 'EventCode']].head()}")
    
    if 'timestamp' in df.columns and 'EventCode' in df.columns:
        print("\n✅ SUCCESS! Ready for analysis")
    else:
        print("\n❌ FAILED")

if __name__ == '__main__':
    main()
