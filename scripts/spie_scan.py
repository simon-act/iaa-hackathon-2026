"""Search for "flags" (21/10, Simone's question): column values that, although rare, signal many more (or fewer) claims.
For each line, column and value (categories; for numerics: missing, 0, min, max and values with share > 2%):
 - raw lift = reported claims / expected at average frequency (exposure × completeness × peril average frequency);
 - stability: lift 2021–22 and 2023–24;
 - freq×sev A/E (OOF 2023–24, own (no spreading) judge): if the model already captures it, A/E ≈ 1;
 - share of the value in the 2025 test.
Output results_s3/spie_scan.csv. Usage: IAA_MARKET=configs/parte1_s3.json python scripts/spie_scan.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
SKIP = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_raw", "target_b",
        "rec_adj", "year", "construction_type_raw"}
J = pd.read_parquet(P.RESULTS / "judge_own.parquet", columns=["row_id", "target_rep", "pi_eff", "ll_expo"]).set_index("row_id")
O = pd.read_parquet(P.RESULTS / "oof/fs2_v2_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]
cust = pd.read_parquet(Path(P.CFG["raw"]) / "customers.parquet", columns=["customer_id", "surname"]) if (Path(P.CFG["raw"]) / "customers.parquet").exists() else None
out = []
for line in P.LINES:
    names = pq.ParquetFile(P.MODEL_DIR / f"model_{line}.parquet").schema.names
    feats = [c for c in names if c not in SKIP and not c.startswith(("n_", "cost_", "excess_", "pi_", "ult"))]
    nc, pc = [c for c in names if c.startswith("n_")], [c for c in names if c.startswith("pi_")]
    perils = [c[2:] for c in nc if f"pi_{c[2:]}" in names]
    d = pd.read_parquet(P.MODEL_DIR / f"model_{line}.parquet", columns=["row_id", "customer_id", "year", "exposure"] + feats + [f"n_{p}" for p in perils] + [f"pi_{p}" for p in perils])
    if cust is not None:
        d = d.merge(cust, on="customer_id", how="left"); feats = feats + ["surname"]
    te_names = pq.ParquetFile(P.MODEL_DIR / f"test_{line}.parquet").schema.names
    te = pd.read_parquet(P.MODEL_DIR / f"test_{line}.parquet", columns=["customer_id", "exposure"] + [c for c in feats if c in te_names])
    if cust is not None:
        te = te.merge(cust, on="customer_id", how="left")
    fp = {p: d[f"n_{p}"].sum() / (d["exposure"] * d[f"pi_{p}"]).sum() for p in perils}
    d["N"] = sum(d[f"n_{p}"] for p in perils); d["E"] = sum(d["exposure"] * d[f"pi_{p}"] * fp[p] for p in perils)
    d["late"] = d["year"] >= 2023
    j = J.reindex(d["row_id"]); pr = O.reindex(d["row_id"]).to_numpy()
    d["mA"] = np.where(np.isfinite(pr), j["target_rep"].to_numpy(), np.nan)
    d["mE"] = np.where(np.isfinite(pr), np.clip(pr - j["ll_expo"].to_numpy(), 1e-3, None) * j["pi_eff"].to_numpy(), np.nan)
    tot_e = d["exposure"].sum(); te_e = te["exposure"].sum()
    for c in feats:
        s = d[c]; st = te[c] if c in te.columns else None
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if (not num) or s.nunique() <= 40:
            key = s.astype("string").fillna("NaN"); kt = st.astype("string").fillna("NaN") if st is not None else None
            top = d.groupby(key, observed=True)["exposure"].sum().sort_values(ascending=False)
            keep = set(top.index[:300]); key = key.where(key.isin(keep), "altro")
            if kt is not None:
                kt = kt.where(kt.isin(keep), "altro")
        else:
            sp = {"NaN": s.isna(), "=0": s == 0, f"=min {s.min():g}": s == s.min(), f"=max {s.max():g}": s == s.max()}
            vc = s.value_counts(normalize=True)
            for v in vc.index[vc > 0.02][:5]:
                sp[f"={v:g}"] = s == v
            key = pd.Series("altro", index=d.index, dtype="string")
            for k, m in sp.items():
                key = key.mask(m.fillna(False) & (key == "altro"), k)
            kt = None
            if st is not None:
                kt = pd.Series("altro", index=te.index, dtype="string")
                spt = {"NaN": st.isna(), "=0": st == 0, f"=min {s.min():g}": st == s.min(), f"=max {s.max():g}": st == s.max()}
                for v in vc.index[vc > 0.02][:5]:
                    spt[f"={v:g}"] = st == v
                for k, m in spt.items():
                    kt = kt.mask(m.fillna(False) & (kt == "altro"), k)
        g = d.groupby(key, observed=True).agg(expo=("exposure", "sum"), N=("N", "sum"), E=("E", "sum"), mA=("mA", "sum"), mE=("mE", "sum"))
        ge = d[~d.late].groupby(key[~d.late], observed=True).agg(N=("N", "sum"), E=("E", "sum"))
        gl = d[d.late].groupby(key[d.late], observed=True).agg(N=("N", "sum"), E=("E", "sum"))
        g["lift"] = g.N / g.E; g["z"] = (g.N - g.E) / np.sqrt(g.E.clip(lower=1e-9))
        g["lift_21_22"] = (ge.N / ge.E).reindex(g.index); g["lift_23_24"] = (gl.N / gl.E).reindex(g.index)
        g["AE_modello"] = g.mA / g.mE; g["quota_esp"] = g.expo / tot_e
        g["quota_test"] = (te.groupby(kt, observed=True)["exposure"].sum() / te_e).reindex(g.index) if kt is not None else np.nan
        g = g.reset_index(names="valore"); g.insert(0, "colonna", c); g.insert(0, "linea", line)
        out.append(g)
    print(line, "done", flush=True)
t = pd.concat(out, ignore_index=True)
t.to_csv(P.RESULTS / "spie_scan.csv", index=False)
f = t[(t.valore != "altro") & (t.quota_esp < 0.05) & (t.expo > 50) & (t.z.abs() > 4) & ((t.lift > 1.5) | (t.lift < 0.67))]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f.sort_values("z", key=abs, ascending=False)[["linea", "colonna", "valore", "expo", "N", "lift", "lift_21_22", "lift_23_24", "AE_modello", "quota_esp", "quota_test", "z"]].round(3).head(80).to_string(index=False))
