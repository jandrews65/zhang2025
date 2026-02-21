# Data Backup Information

**Last Backup:** 2026-02-21 19:10:55

## Backed Up to OneDrive

**Location:** C:\Users\Jpand\OneDrive\zhang2025-backup-2026-02-21

### Critical Data (Cannot Easily Recreate)
- **Sentiment Scores:** /app/data/sentiment/ (~10GB)
  - 303,625 files, 96 hours processing on GTX 1660
  - Partitioned by year/month

### Important Data (Hours to Recreate)  
- **Headlines:** /app/data/headlines/ (~8GB)
  - 140M+ extracted headlines, 3.5 hours processing
  
- **GDELT Data:** /app/data/gdelt/ (~5GB)
  - Filtered event files, 2 hours to re-download

### Generated Data (Minutes to Recreate)
- **Features:** /app/data/processed/ (<1GB)
  - Can regenerate in 30 minutes from sentiment scores

## Restore Instructions

\\\powershell
# Full restore
docker cp "C:\Users\Jpand\OneDrive\zhang2025-backup-2026-02-21\sentiment" zhang2025:/app/data/
docker cp "C:\Users\Jpand\OneDrive\zhang2025-backup-2026-02-21\headlines" zhang2025:/app/data/
docker cp "C:\Users\Jpand\OneDrive\zhang2025-backup-2026-02-21\gdelt" zhang2025:/app/data/
docker cp "C:\Users\Jpand\OneDrive\zhang2025-backup-2026-02-21\processed" zhang2025:/app/data/
\\\

## Backup Size
- Sentiment: ~10GB
- Headlines: ~8GB  
- GDELT: ~5GB
- Processed: <1GB
- **Total: ~24GB**

