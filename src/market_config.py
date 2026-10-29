"""Per-market configuration (Part 1 = default; Part 2 = a JSON file in configs/).

Design choice: the defaults below are EXACTLY those used in Part 1, so without a config
the pipeline produces the same numbers as before. A new market is described in configs/<name>.json and activated with
    IAA_MARKET=configs/parte2.json python -I src/clean_policies.py ...
or with run_all.sh configs/parte2.json. Keys missing from the JSON take the Part 1 default.

Why a file and not scattered parameters: the judges assess "how the model adapts"; the difference between the two markets
must live in one readable place, and the code must stay identical.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = {
    "name": "parte1",
    "raw": "data/raw",                   # raw tables (policies_<line>, test_<line>, claims, customers, addresses, weather)
    "interim": "data/interim",
    "model_dir": "data/model",
    "results": "results",
    "lines": ["motor", "buildings", "contents"],
    "target_year": 2025,                 # year to predict
    "train_years": None,                 # None = all years present in the policies
    "folds": None,                       # None = last two years as validation, e.g. {"F2023": [[2021,2022],2023], ...}
    "holdout_share": 0.10,
    "cap": {"motor": 150000.0, "buildings": 250000.0, "contents": 50000.0},   # large-loss cap by line
    "weather_perils": [["buildings", "subsidence"], ["buildings", "storm"], ["buildings", "flood"],
                       ["contents", "flood"], ["motor", "glass_hail"]],
    "declined_status": ["repudiated", "declined", "rejected", "denied"],   # claims excluded from the target
    "rename": {},                        # new-market columns -> Part 1 name, e.g. {"no_claims_years": "ncb_years"}
    "fixes": {                           # Part 1 data corrections: can be switched off
        "customer_id_zfill": 8,          # 0 = off
        "sum_insured_div_if_per_bedroom_over": 100.0,   # 0 = off
        "construction_map": {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "w": "timber", "t": "timber"},
        "exposure_from_dates": True,
        "scale": {},                     # Part 2: columns to convert to Part 1 units, e.g. {"sum_insured": 1000}
        "loss_date_fix": False           # 18/10: True (or threshold in days) = loss date from report date − median delay
    },
    "trend": {
        "sev_nonsig": "line_median"      # severity with non-significant trend: "line_median" (Part 1 until 18/10), "own" (own estimate), "zero"
    },
    "transfer": {                        # Part 2: adaptation from the source market
        "source_market": None,           # e.g. "configs/parte1.json": None = no transfer
        "mode": "offset_credibility"     # see src/transfer.py
    }
}


def _merge(d, u):
    out = dict(d)
    for k, v in u.items():
        out[k] = _merge(d[k], v) if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    return out


def load(path: str | None = None) -> dict:
    path = path or os.environ.get("IAA_MARKET")
    cfg = DEFAULT
    if path:
        cfg = _merge(DEFAULT, json.loads(Path(path).read_text()))
        cfg["_path"] = str(path)
    for k in ("raw", "interim", "model_dir", "results"):
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else ROOT / p)
    cfg["cap"] = {k: float(v) for k, v in cfg["cap"].items()}
    cfg["weather_perils"] = {tuple(x) for x in cfg["weather_perils"]}
    return cfg


CFG = load()
LINES = CFG["lines"]
TARGET_YEAR = int(CFG["target_year"])


def folds_from_years(years) -> dict:
    """time folds: each year from the third onwards is evaluated with the fit on previous years; with 3 years → 1 fold."""
    years = sorted(set(int(y) for y in years))
    if CFG["folds"]:
        return {k: (list(v[0]), int(v[1])) for k, v in CFG["folds"].items()}
    out = {}
    for i in range(max(1, len(years) - 2), len(years)):
        out[f"F{years[i]}"] = (years[:i], years[i])
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(CFG)
