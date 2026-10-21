"""Configurazione per mercato (Parte 1 = default; Parte 2 = un file JSON in configs/).

Scelta: i valori di default qui sotto sono ESATTAMENTE quelli usati nella Parte 1, così senza config
la catena produce gli stessi numeri di prima. Un mercato nuovo si descrive in configs/<nome>.json e si attiva con
    IAA_MARKET=configs/parte2.json python -I src/clean_policies.py ...
oppure con run_all.sh configs/parte2.json. Le chiavi assenti nel JSON prendono il default della Parte 1.

Perché un file e non parametri sparsi: i giudici valutano "come il modello si adatta"; la differenza tra i due mercati
deve stare in un posto solo, leggibile, e il codice deve restare identico.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = {
    "name": "parte1",
    "raw": "data/raw",                   # tabelle grezze (policies_<linea>, test_<linea>, claims, customers, addresses, weather)
    "interim": "data/interim",
    "model_dir": "data/model",
    "results": "results",
    "lines": ["motor", "buildings", "contents"],
    "target_year": 2025,                 # anno da prevedere
    "train_years": None,                 # None = tutti gli anni presenti nelle polizze
    "folds": None,                       # None = ultimi due anni come validazione, es. {"F2023": [[2021,2022],2023], ...}
    "holdout_share": 0.10,
    "cap": {"motor": 150000.0, "buildings": 250000.0, "contents": 50000.0},   # taglio grandi sinistri per linea
    "weather_perils": [["buildings", "subsidence"], ["buildings", "storm"], ["buildings", "flood"],
                       ["contents", "flood"], ["motor", "glass_hail"]],
    "declined_status": ["repudiated", "declined", "rejected", "denied"],   # sinistri esclusi dal target
    "rename": {},                        # colonne del mercato nuovo -> nome della Parte 1, es. {"no_claims_years": "ncb_years"}
    "fixes": {                           # correzioni dati della Parte 1: si possono spegnere
        "customer_id_zfill": 8,          # 0 = spento
        "sum_insured_div_if_per_bedroom_over": 100.0,   # 0 = spento
        "construction_map": {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "w": "timber", "t": "timber"},
        "exposure_from_dates": True,
        "scale": {},                     # Parte 2: colonne da riportare alle unità della Parte 1, es. {"sum_insured": 1000}
        "loss_date_fix": False           # 18/10: True (o soglia in giorni) = data di accadimento da denuncia − ritardo mediano
    },
    "trend": {
        "sev_nonsig": "line_median"      # severità con trend non significativo: "line_median" (Parte 1 fino al 18/10), "own" (stima propria), "zero"
    },
    "transfer": {                        # Parte 2: adattamento dal mercato di partenza
        "source_market": None,           # es. "configs/parte1.json": None = nessun trasferimento
        "mode": "offset_credibility"     # vedi src/transfer.py
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
    """fold temporali: ogni anno dal terzo in poi è valutato con la stima sugli anni precedenti; con 3 anni → 1 fold."""
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
