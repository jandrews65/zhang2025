"""
GDELT Column Inspector - See what we're actually getting
"""

import io
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "http://data.gdeltproject.org/gdeltv2"

# Try to download one recent file
end_date = datetime.now()
timestamp = end_date.strftime('%Y%m%d') + "000000"
url = f"{BASE_URL}/{timestamp}.export.CSV.zip"

print(f"Downloading test file: {url}")

try:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    print("✓ Downloaded successfully")
except Exception as e:
    print(f"✗ Download failed: {e}")
    exit(1)

# Extract
try:
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
        print(f"✓ Found CSV: {csv_names[0]}")
        csv_data = zf.read(csv_names[0])
except Exception as e:
    print(f"✗ Extract failed: {e}")
    exit(1)

# Read without column specifications to see what we get
print("\nInspecting raw CSV structure...")

try:
    # Read first few lines to count columns
    lines = csv_data.decode('latin-1').split('\n')[:5]
    
    for i, line in enumerate(lines):
        if line.strip():
            cols = line.split('\t')
            print(f"\nLine {i+1}: {len(cols)} columns")
            print(f"First 10 columns: {cols[:10]}")
            print(f"Columns 24-30 (EventCode area): {cols[24:30] if len(cols) > 30 else 'N/A'}")
    
    # Try reading with pandas
    df = pd.read_csv(
        io.BytesIO(csv_data),
        sep="\t",
        header=None,
        nrows=5,
        encoding='latin-1',
        on_bad_lines='skip'
    )
    
    print(f"\n✓ Pandas loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"\nFirst row sample:")
    print(df.iloc[0, :10].to_dict())
    
    # Check EventCode position (should be column 26, 0-indexed = 26)
    if len(df.columns) > 26:
        print(f"\nColumn 26 (EventCode): {df.iloc[:, 26].tolist()}")
    else:
        print(f"\n✗ Only {len(df.columns)} columns, expected 61+")
    
except Exception as e:
    print(f"✗ Inspection failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("Based on column count, update GDELT_COLUMNS list to match actual format")
