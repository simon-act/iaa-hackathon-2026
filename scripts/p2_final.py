"""Parte 2: submission del mercato emerging stimata su tutti i dati nuovi (2023+2024) e il prior del developed.
Specifica per linea (scelta sul fold F2024, scripts/p2_sweep2.py; docs/30_parte2.md):
  motor     : prior developed × livello emerging × correzione genere (GLM ridge, offset = prior)
  buildings : 0,9 × prior × livello + 0,1 × scratch (freq×sev solo emerging), stesso livello
  contents  : prior × livello × correzione genere
Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_final.py <nome> [--spec base|...]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "v1": {"motor": ("gender", 0.0), "buildings": ("level", 0.1), "contents": ("gender", 0.0)},
    "v2": {"motor": ("gender", 0.1), "buildings": ("level", 0.1), "contents": ("gender", 0.0)},   # + 10% scratch in auto (simulazione)
    "nonb": None,
    # 24/10 sera: auto con correzione nuovo business stimata sull'emerging 2023+24 (A/E nuovo business 1,18 contro 0,98 dei rinnovi)
    "v3": {"motor": ("gender+nb", 0.0), "buildings": ("level", 0.1), "contents": ("gender", 0.0)},
    "level": {"motor": ("level", 0.0), "buildings": ("level", 0.0), "contents": ("level", 0.0)},
    # 25/10: contenuto con livello per perilo (credibilità k 100/200 verso la linea) e genere sopra: banco 7 verifiche +8,1/+7,0, 6/7
    "v4": {"motor": ("gender+nb", 0.0), "buildings": ("level", 0.1), "contents": ("gender_peril", 0.0)},
    # 25/10 sera: varianti del nuovo business in auto (decisione di Simone): nessuna correzione / correzione stimata su target tagliato a 25k
    "v5": {"motor": ("gender", 0.0), "buildings": ("level", 0.1), "contents": ("gender_peril", 0.0)},
    "v6": {"motor": ("gender+nb_cap25k", 0.0), "buildings": ("level", 0.1), "contents": ("gender_peril", 0.0)},
}
name = sys.argv[1]
spec = SPECS[sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "v1"]
if "--prior" in sys.argv:   # 24/10: prior alternativo, es. --prior _meteo (estrapolazione del meteo, scripts/p2_weather.py)
    TR.PRIOR_SUFFIX = sys.argv[sys.argv.index("--prior") + 1]
out, log = [], {}
for l in P.LINES:
    tr, te = P.load(l), P.load(l, "test")
    kind, w = spec[l]
    if kind == "gender_peril":
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_te = np.asarray(TR.src_peril(tr, te, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        TR._PRIOR_OVERRIDE = pd.DataFrame({"mu_src": np.r_[p_tr, p_te]}, index=np.r_[tr.row_id.to_numpy(), te.row_id.to_numpy()])
        TR.CORRECTION_VARS, old = ["gender"], TR.CORRECTION_VARS
        p = TR.src_glm(tr, te, l); TR.CORRECTION_VARS = old; TR._PRIOR_OVERRIDE = None
        log[l] = {"glm": str(TR.src_glm.last), "livelli_perilo": str(TR.src_peril.last_log)}
    elif kind == "gender+nb_cap25k":
        trc = tr.copy(); trc[P.TARGET] = trc[P.TARGET].clip(upper=25000)
        TR.CORRECTION_VARS, old = ["new_business", "gender"], TR.CORRECTION_VARS
        p = np.asarray(TR.src_glm(trc, te, l), float) * tr[P.TARGET].sum() / trc[P.TARGET].sum()
        TR.CORRECTION_VARS = old; log[l] = {"glm": str(TR.src_glm.last), "nota": "GLM stimato su target tagliato a 25k, livello del target pieno"}
    elif kind in ("gender", "gender+nb"):
        TR.CORRECTION_VARS, old = (["gender"] if kind == "gender" else ["new_business", "gender"]), TR.CORRECTION_VARS
        p = TR.src_glm(tr, te, l); TR.CORRECTION_VARS = old; log[l] = {"glm": str(TR.src_glm.last)}
    else:
        p = TR.src_level(tr, te, l); log[l] = {"k": TR.src_level.last_k}
    if w > 0:
        sc = np.asarray(TR.scratch(tr, te, l), float)
        p = (1 - w) * p + w * sc * p.sum() / sc.sum()
    log[l]["tasso"] = float(p.sum() / te["exposure"].sum()); log[l]["target_train_tasso"] = float(tr[P.TARGET].sum() / tr["exposure"].sum())
    print(l, log[l], flush=True)
    out.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": p}))
o = pd.concat(out)
ss = pd.read_parquet(Path(P.CFG["raw"]) / "sample_submission.parquet", columns=["row_id"])
o = ss.merge(o, on="row_id", how="left")
assert o.prediction.notna().all() and (o.prediction >= 0).all() and len(o) == len(ss)
o["prediction"] = o.prediction.round(3)
o.to_csv(ROOT / "submissions_p2" / f"{name}.csv", index=False)
(ROOT / "submissions_p2" / f"{name}.json").write_text(json.dumps(log, indent=1, default=str))
print("->", ROOT / "submissions_p2" / f"{name}.csv", len(o))
