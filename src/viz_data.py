"""Aggrega i numeri per la pagina di visualizzazione del passo 1 (un solo JSON piccolo).
Uso: python -I viz_data.py <raw> <interim> <model> <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw, interim, model, out = map(Path, sys.argv[1:5])
J = {}
LINES = ["motor", "buildings", "contents"]

# --- pulizia sinistri
rc = json.loads((interim / "report_clean_claims.json").read_text())
J["claims_steps"] = {"raw": rc["claims_raw"], "steps": rc["steps"], "clean": rc["claims_clean"]}
rp = json.loads((interim / "report_clean_policies.json").read_text())
J["exposure_fix"] = {l: {"original": rp[f"train_{l}"]["exposure_sum_original"],
                         "clean": rp[f"train_{l}"]["exposure_sum_clean"],
                         "rows_fixed": rp[f"train_{l}"]["exposure_fixed"],
                         "days_coded": rp[f"train_{l}"]["exposure_fixed_days_coded"],
                         "r365_360": rp[f"train_{l}"]["exposure_fixed_365_360"]} for l in LINES}
J["policy_fixes"] = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, int)} for k, v in rp.items()}

# --- construction W/T evidence
tr = pd.read_parquet(interim / "train_buildings.parquet", columns=["construction_type_raw"])
te = pd.read_parquet(interim / "test_buildings.parquet", columns=["construction_type_raw"])
def codes(s):
    s = s.astype(str).str.strip()
    lab = s.str.lower().map({"brick": "brick", "stone": "stone", "non_standard": "non_standard"}).fillna(s.str.upper())
    return (lab.value_counts(normalize=True) * 100).round(2).to_dict()
J["construction"] = {"train": codes(tr["construction_type_raw"]), "test": codes(te["construction_type_raw"])}

# --- customer id lengths (motor)
for k in ("policies", "test"):
    t = pd.read_parquet(raw / f"{k}_motor.parquet", columns=["customer_id"])
    J.setdefault("custid_len", {})[k] = t["customer_id"].str.len().value_counts().sort_index().to_dict()

# --- reporting curves + completeness by month
rd = json.loads((interim / "report_development.json").read_text())
J["report_cdf"] = {"days": [0, 7, 14, 30, 60, 90, 180, 365, 730, 1095, 1460],
                   "curves": rd["reporting_cdf_at_days[0,7,14,30,60,90,180,365,730,1095,1460]"]}
c = pd.read_parquet(interim / "claims_clean.parquet")
# completezza per mese di accadimento: media sulle righe? usiamo F direttamente via develop: ricalcolo per mese
sys.path.insert(0, str(Path(__file__).parent))
from develop import reporting_cdf, completeness_by_day, MAX_LAG  # noqa: E402
surf = {"months": [], "perils": [], "z": []}
months = pd.period_range("2021-01", "2024-12", freq="M")
surf["months"] = [str(m) for m in months]
for (line, peril), g in c.groupby(["line", "peril"]):
    F = reporting_cdf(g["report_lag_days"].clip(0, MAX_LAG).to_numpy(), g["days_to_valuation"].clip(0, MAX_LAG + 1).to_numpy())
    cd = completeness_by_day(F)
    z = cd.groupby(cd.index.to_period("M")).mean().reindex(months).round(4).tolist()
    surf["perils"].append(f"{line} · {peril}")
    surf["z"].append(z)
J["completeness_surface"] = surf

# --- sviluppo
d = pd.read_parquet(interim / "development_onlevel.parquet")
d["ibner_amt"] = d["n_rep"] * (d["sev_rep_ult"] - d["sev_rep"])
d["ibnr_amt"] = d["cost_ult"] - d["n_rep"] * d["sev_rep_ult"]
J["development"] = d[["line", "peril", "ay", "exposure", "n_rep", "n_ult", "open_share", "pi", "cost_rep", "ibner_amt",
                      "ibnr_amt", "cost_ult", "freq_ult", "sev_ult", "sev_rep", "sev_prior", "dev_factor",
                      "sev_onlevel", "freq_onlevel", "bc_ult", "bc_onlevel"]].round(5).to_dict(orient="records")
t = pd.read_parquet(interim / "trend.parquet")
J["trend"] = t[["line", "peril", "freq_trend", "freq_rule", "sev_trend", "sev_rule", "freq_2025", "sev_2025", "bc_2025"]].round(5).to_dict(orient="records")
lv = pd.read_parquet(interim / "level_2025.parquet")
J["level_2025"] = lv.round(3).reset_index().to_dict(orient="records")

# --- large loss: istogramma log10 del costo netto per linea
bins = np.arange(0, 7.21, 0.1)
J["claim_size_hist"] = {"bins": bins[:-1].round(2).tolist(), "cap": {"motor": 150000, "buildings": 250000, "contents": 50000}}
for line, g in c.groupby("line"):
    v = np.log10(g["net_incurred"].clip(lower=1))
    J["claim_size_hist"][line] = np.histogram(v, bins=bins)[0].tolist()
J["largest"] = c.nlargest(8, "net_incurred")[["claim_id", "line", "peril", "status", "ay", "net_incurred", "net_paid"]].round(0).to_dict(orient="records")

# --- meteo vs subsidence / hail
w = pd.read_parquet(raw / "weather.parquet")
wy = w.groupby("year")[["hail_days", "rainfall_mm", "wind_exposure_index", "dry_summer_index", "freezing_days", "mean_temperature_c"]].mean().round(4)
J["weather_by_year"] = wy.reset_index().to_dict(orient="records")
sub = d[(d["line"] == "buildings") & (d["peril"].isin(["subsidence", "storm", "flood"]))][["peril", "ay", "freq_ult"]]
sub = pd.concat([sub, d[(d["line"] == "motor") & (d["peril"] == "glass_hail")][["peril", "ay", "freq_ult"]]])
J["weather_perils"] = sub.round(6).to_dict(orient="records")

# --- 3D: flood buildings per zona × distanza dal fiume; subsidence per suolo × siccità
mb = pd.read_parquet(model / "model_buildings.parquet",
                     columns=["exposure", "n_flood", "pi_flood", "flood_zone", "distance_to_river_km",
                              "n_subsidence", "pi_subsidence", "soil_type", "dry_summer_index", "trees_within_5m"])
mb["dist_b"] = pd.cut(mb["distance_to_river_km"], [0, 0.5, 1, 2, 4, 8, 31], labels=["<0,5", "0,5–1", "1–2", "2–4", "4–8", ">8"])
fz = mb.dropna(subset=["flood_zone"]).groupby(["flood_zone", "dist_b"], observed=True).apply(
    lambda g: g["n_flood"].sum() / (g["exposure"] * g["pi_flood"]).sum() * 1000 if g["exposure"].sum() > 2000 else np.nan,
    include_groups=False).unstack()
J["flood_surface"] = {"zones": [int(z) for z in fz.index], "dist": [str(x) for x in fz.columns], "z": [[None if pd.isna(v) else round(float(v), 3) for v in r] for r in fz.values]}
mb["dry_b"] = pd.cut(mb["dry_summer_index"], [0, .3, .4, .5, .6, .7, 1.0], labels=["<0,3", "0,3–0,4", "0,4–0,5", "0,5–0,6", "0,6–0,7", ">0,7"])
ss = mb[mb["soil_type"] != "missing"].groupby(["soil_type", "dry_b"], observed=True).apply(
    lambda g: g["n_subsidence"].sum() / (g["exposure"] * g["pi_subsidence"]).sum() * 1000 if g["exposure"].sum() > 2000 else np.nan,
    include_groups=False).unstack()
J["subs_surface"] = {"soil": list(ss.index), "dry": [str(x) for x in ss.columns], "z": [[None if pd.isna(v) else round(float(v), 3) for v in r] for r in ss.values]}

# --- target (b)
tb = {}
for line in LINES:
    m = pd.read_parquet(model / f"model_{line}.parquet", columns=["target_b", "exposure", "n_total", "year"])
    rate = m["target_b"] / m["exposure"]
    pos = m["target_b"][m["n_total"] > 0]
    tb[line] = {"rows": len(m), "share_with_claim": round(float((m["n_total"] > 0).mean()), 4),
                "mean_per_expo": round(float(m["target_b"].sum() / m["exposure"].sum()), 2),
                "hist_log10_pos": np.histogram(np.log10(pos.clip(lower=1)), bins=bins)[0].tolist(),
                "zero_rows_mean": round(float(m.loc[m["n_total"] == 0, "target_b"].mean()), 2),
                "by_year": m.groupby("year").apply(lambda g: g["target_b"].sum() / g["exposure"].sum(), include_groups=False).round(2).to_dict()}
J["target_b"] = tb
sub0 = pd.read_csv(Path(__file__).resolve().parents[1] / "submissions" / "sub_00_pipeline_test_line_level.csv")
J["submission_test"] = {"rows": len(sub0), "mean": round(float(sub0["prediction"].mean()), 2)}
out.write_text(json.dumps(J, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("ok", out.stat().st_size // 1024, "KB")
