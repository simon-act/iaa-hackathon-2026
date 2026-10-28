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
    # 26/10: v4 + fabbricato con livello proprio per storm (k 100/200), resto al livello della linea; banco p2_bench10 5/7 forma e own
    # 26/10 sera: SONDA (non candidata, bench11 4/7 forma): v4 + fuel_type e cover_type nella correzione auto
    "v8": {"motor": ("gender+nb+fuel+cover", 0.0), "buildings": ("level", 0.1), "contents": ("gender_peril", 0.0)},
    # 27/10: v7 (storm) + auto 50/50 tra la correzione di sub_10 e quella con fuel_type e cover_type; banco p2_bench12 6/7 forma e own
    "v9": {"motor": ("half_fuel_cover", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril", 0.0)},
    # 27/10: v9 + contenuto × correzione di frequenza works_from_home (GLM Poisson su numeri, offset prior); banco p2_bench13 6/7
    "v10": {"motor": ("half_fuel_cover", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    # 27/10: v10 + auto × GBM Poisson sul numero di sinistri (alberi con lgb.cv); banco p2_bench14 +8,9 / +6,3, 5/7 (F2024 +34,5)
    "v11": {"motor": ("half_fuel_cover_gbmfreq", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    # 27/10: v11 + premio relativo in auto (idea di Simone dal modello migliore sul privato P1, fs2_v2_cred): prem_rel tra le variabili
    # del GBM sul numero di sinistri e correzione di frequenza su prem_rel; banco p2_bench17b +18,2 / +11,7, 7/7 · 6/7 contro sub_18
    "v12": {"motor": ("half_fuel_cover_gbmfreq_prem", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    # 27/10: v12 con il prior del modello migliore sul privato P1 (fs2_v2 = premio relativo nel freq×sev del developed) in fabbricato
    # e contenuto (source_pred_<linea>_p1v2, scripts/p2_export_p1.py); banco p2_bench19 +7,5 / +6,8 e +6,4 / +5,9, 5/7
    "v13": {"motor": ("half_fuel_cover_gbmfreq_prem", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    # 28/10: SONDE = v13 con il prior auto mediato su due metà del developed (bagging), con e senza premio relativo nel prior
    "v14": {"motor": ("half_fuel_cover_gbmfreq_prem", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    "v15": {"motor": ("half_fuel_cover_gbmfreq_prem", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril_freqwfh", 0.0)},
    "v7": {"motor": ("gender+nb", 0.0), "buildings": ("level_storm", 0.1), "contents": ("gender_peril", 0.0)},
}
name = sys.argv[1]
spec = SPECS[sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "v1"]
if "--prior" in sys.argv:   # 24/10: prior alternativo, es. --prior _meteo (estrapolazione del meteo, scripts/p2_weather.py)
    TR.PRIOR_SUFFIX = sys.argv[sys.argv.index("--prior") + 1]
out, log = [], {}
PRIOR_BY_LINE = {"v13": {"buildings": "_p1v2", "contents": "_p1v2"},
                 "v14": {"motor": "_blbagv2prem", "buildings": "_p1v2", "contents": "_p1v2"},
                 "v15": {"motor": "_blbagv2", "buildings": "_p1v2", "contents": "_p1v2"}}   # 27/10: prior diverso per linea
_spec_name = sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "v1"
_base_suffix = TR.PRIOR_SUFFIX
for l in P.LINES:
    TR.PRIOR_SUFFIX = PRIOR_BY_LINE.get(_spec_name, {}).get(l, _base_suffix)
    tr, te = P.load(l), P.load(l, "test")
    kind, w = spec[l]
    if kind.endswith("_prem"):   # 27/10: premio relativo (log premio lordo di sconto − mediana dell'anno), come in fs2_v2_cred
        tr, te, kind = M._prem(tr), M._prem(te), kind[:-5]
        prem = True
    else:
        prem = False
    if kind in ("gender_peril", "gender_peril_freqwfh"):
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_te = np.asarray(TR.src_peril(tr, te, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        TR._PRIOR_OVERRIDE = pd.DataFrame({"mu_src": np.r_[p_tr, p_te]}, index=np.r_[tr.row_id.to_numpy(), te.row_id.to_numpy()])
        TR.CORRECTION_VARS, old = ["gender"], TR.CORRECTION_VARS
        p = TR.src_glm(tr, te, l); TR.CORRECTION_VARS = old; TR._PRIOR_OVERRIDE = None
        log[l] = {"glm": str(TR.src_glm.last), "livelli_perilo": str(TR.src_peril.last_log)}
        if kind == "gender_peril_freqwfh":
            p = np.asarray(p, float) * TR.freq_correction(tr, te, l, ["works_from_home"]); log[l]["freq_wfh"] = str(TR.freq_correction.last)
    elif kind == "gender+nb_cap25k":
        trc = tr.copy(); trc[P.TARGET] = trc[P.TARGET].clip(upper=25000)
        TR.CORRECTION_VARS, old = ["new_business", "gender"], TR.CORRECTION_VARS
        p = np.asarray(TR.src_glm(trc, te, l), float) * tr[P.TARGET].sum() / trc[P.TARGET].sum()
        TR.CORRECTION_VARS = old; log[l] = {"glm": str(TR.src_glm.last), "nota": "GLM stimato su target tagliato a 25k, livello del target pieno"}
    elif kind in ("half_fuel_cover", "half_fuel_cover_gbmfreq"):
        TR.CORRECTION_VARS, old = ["new_business", "gender"], TR.CORRECTION_VARS
        p1 = np.asarray(TR.src_glm(tr, te, l), float); g1 = str(TR.src_glm.last)
        TR.CORRECTION_VARS = ["new_business", "gender", "fuel_type", "cover_type"]
        p2 = np.asarray(TR.src_glm(tr, te, l), float); TR.CORRECTION_VARS = old
        p = 0.5 * p1 + 0.5 * p2; log[l] = {"glm_sub10": g1, "glm_fuel_cover": str(TR.src_glm.last)}
        if kind == "half_fuel_cover_gbmfreq":
            TR.CORRECTION_VARS = ["new_business", "gender"]; q1 = np.asarray(TR.src_glm(tr, tr, l), float)
            TR.CORRECTION_VARS = ["new_business", "gender", "fuel_type", "cover_type"]; q2 = np.asarray(TR.src_glm(tr, tr, l), float)
            TR.CORRECTION_VARS = old
            p = p * TR.gbm_freq_correction(tr, te, l, 0.5 * q1 + 0.5 * q2); log[l]["gbm_freq"] = str(TR.gbm_freq_correction.last)
            if prem:
                p = p * TR.freq_correction(tr, te, l, ["prem_rel"]); log[l]["freq_prem"] = str(TR.freq_correction.last)
    elif kind == "gender+nb+fuel+cover":
        TR.CORRECTION_VARS, old = ["new_business", "gender", "fuel_type", "cover_type"], TR.CORRECTION_VARS
        p = TR.src_glm(tr, te, l); TR.CORRECTION_VARS = old; log[l] = {"glm": str(TR.src_glm.last)}
    elif kind in ("gender", "gender+nb"):
        TR.CORRECTION_VARS, old = (["gender"] if kind == "gender" else ["new_business", "gender"]), TR.CORRECTION_VARS
        p = TR.src_glm(tr, te, l); TR.CORRECTION_VARS = old; log[l] = {"glm": str(TR.src_glm.last)}
    elif kind == "level_storm":
        ov, r, lg = TR.peril_own_override(tr, te, l, ["storm"], 100.0, 200.0)
        TR._PRIOR_OVERRIDE = ov; p = TR.src_level(tr, te, l); TR._PRIOR_OVERRIDE = None
        log[l] = {"k": TR.src_level.last_k, "r_perilo": {k: round(v, 4) for k, v in r.items()}, "livelli_perilo": str(lg)}
    else:
        p = TR.src_level(tr, te, l); log[l] = {"k": TR.src_level.last_k}
    if w > 0:
        sc = np.asarray(TR.scratch(tr, te, l), float)
        p = (1 - w) * p + w * sc * p.sum() / sc.sum()
    log[l]["tasso"] = float(p.sum() / te["exposure"].sum()); log[l]["target_train_tasso"] = float(tr[P.TARGET].sum() / tr["exposure"].sum())
    print(l, log[l], flush=True)
    out.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": p}))
TR.PRIOR_SUFFIX = _base_suffix
o = pd.concat(out)
ss = pd.read_parquet(Path(P.CFG["raw"]) / "sample_submission.parquet", columns=["row_id"])
o = ss.merge(o, on="row_id", how="left")
assert o.prediction.notna().all() and (o.prediction >= 0).all() and len(o) == len(ss)
o["prediction"] = o.prediction.round(3)
o.to_csv(ROOT / "submissions_p2" / f"{name}.csv", index=False)
(ROOT / "submissions_p2" / f"{name}.json").write_text(json.dumps(log, indent=1, default=str))
print("->", ROOT / "submissions_p2" / f"{name}.csv", len(o))
