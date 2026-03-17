"""

PATHS.PY — Central path configuration for the Process Mining Project

Usage:
    from paths import PATHS
    df.to_csv(PATHS['processed'] / 'event_log_cleaned.csv')

"""

from pathlib import Path

# ── Project root — everything is relative to this ──────────────────────────
# This file lives in scripts/, so root is one level up.
ROOT = Path("Process_Mining_Project").resolve().parent.parent

# ── Folder definitions ──────────────────────────────────────────────────────
PATHS = {
    # Raw source data (XES file goes here)
    'raw':        ROOT / 'data' / 'raw',

    # Cleaned, transformed data ready for analysis
    'processed':  ROOT / 'data' / 'processed',

    # All scripts (this folder)
    'scripts':    ROOT / 'scripts',

    # Charts and figures
    'figures':    ROOT / 'results' / 'figures',

    # CSV result tables (for Power BI, Excel, sharing)
    'tables':     ROOT / 'results' / 'tables',

    # Text reports and CV bullet point files
    'reports':    ROOT / 'results' / 'reports',

    # Power BI files (.pbix, DAX snippets)
    'powerbi':    ROOT / 'powerbi',

    # Project root itself
    'root':       ROOT,
}

def create_all_dirs():
   
    for name, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)

# Auto-create on import
create_all_dirs()

if __name__ == '__main__':
    print("Project folder structure:")
    print(f"  Root: {ROOT}\n")
    for name, path in PATHS.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  [{exists}] {name:<12} → {path.relative_to(ROOT)}")
