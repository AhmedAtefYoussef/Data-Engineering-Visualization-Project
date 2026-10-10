"""Central configuration. Single source of truth for IDs, windows, paths and team."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE = os.environ.get("NYC_FIXTURE", "0") == "1"          # synthetic data mode (tests/dev only)

HOUSING_ID = "hg8x-zxpr"   # Affordable Housing Production by Building
CRASH_ID = "h9gi-nx95"     # Motor Vehicle Collisions - Crashes
SODA_URL = "https://data.cityofnewyork.us/resource/{}.csv"
DATASET_PAGES = {
    "housing": "https://data.cityofnewyork.us/d/" + HOUSING_ID,
    "crashes": "https://data.cityofnewyork.us/d/" + CRASH_ID,
}

Y_START, Y_END = 2016, 2025          # analysis window (full calendar years only: latest months lag)
HOUSING_MIN_YEAR = 2014              # earliest completion kept for cumulative units
BUFFER_M = 250                       # spatial exposure radius (metres)
EVENT_WINDOW = 3                     # years before/after completion
GRID_DEG = 0.005                     # map aggregation cell
ZIP_GRID_DEG = 0.002                 # ZIP imputation cell
MIN_ZIP_CRASHES = 50
MIN_N = 8                            # suppress statistics below this many analysis units
SEED = 42
BOOT_PIPELINE, BOOT_APP = 500, 200
NYC_BBOX = dict(lat=(40.45, 40.95), lon=(-74.30, -73.65))
NYC_ZIP_RE = r"^(10[0-4]\d{2}|11[1-4]\d{2}|1100[45]|116\d{2})$"
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

RAW_DIR = ROOT / "data" / "raw"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
PROCESSED_DIR = Path(os.environ.get("NYC_PROCESSED_DIR") or
                     ROOT / "data" / ("processed_fixture" if FIXTURE else "processed"))
APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN")

# Fill in real names/IDs before submission; RQ ids map to members (2 each; 5 members = 10 RQs).
TEAM = [
    {"member": "M1", "name": "TODO Name 1", "student_id": "TODO", "rqs": [1, 2]},
    {"member": "M2", "name": "TODO Name 2", "student_id": "TODO", "rqs": [3, 4]},
    {"member": "M3", "name": "TODO Name 3", "student_id": "TODO", "rqs": [5, 6]},
    {"member": "M4", "name": "TODO Name 4", "student_id": "TODO", "rqs": [7, 8]},
    {"member": "M5", "name": "TODO Name 5", "student_id": "TODO", "rqs": [9, 10]},
]
N_RQ = 10  # 5-member team: 10 research questions + 10 charts
