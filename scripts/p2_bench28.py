"""Part 2 (28/10, A3): single developed + emerging model WITH the relative premium (prem_rel = log premium before discount − median
of the year, computed within each market), vs p2_sub_21, on the honest bench of p2_bench27 (time × customers, 6 checks).
bench9 of 25/10 (without premium) had lost: motor −36.7, buildings −124.6, contents +13.7 (4/7). GBM Tweedie 1.5 on cost/exposure,
exposure weight; developed harmonised to 30% of policies (time limit); market flag; prediction rescaled to the train level.
Variants: 'unico additivo' (1-split trees, 800 rounds) and 'unico interazioni' (15 leaves, 500 rounds).
One fold per call: python3 scripts/p2_bench28.py <line> <fold index 0-5>; with 'riepilogo' instead of the index it prints the results.
Rule from p2_bench27: mean > 0 on shape and own, ≥ 4/6 on both, mean > 0 in both directions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M, train as T  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]; l = sys.argv[1]; OUT = Path(P.RESULTS) / f"p2_bench28_{l}.csv"
PAR = {"unico additivo": dict(num_leaves=2, max_depth=1, learning_rate=0.1, rounds=800),
       "unico interazioni": dict(num_leaves=15, learning_rate=0.05, rounds=500)}


def folds27(df):
    g = pd.util.hash_array(df["customer_id"].astype(str).to_numpy() + "_c27") % 3; yr = df["year"].to_numpy()
    for a, b in ((2023, 2024), (2024, 2023)):
        for k in range(3):
            yield f"{a}→{b} c{k}", df[(yr == a) & (g != k)], df[(yr == b) & (g == k)]


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def sub21(tr, va):
    if l == "motor":
        a = lambda d: 0.5 * glm(tr, d, ["new_business", "gender", "prem_rel"]) + 0.5 * glm(tr, d, ["new_business", "gender", "fuel_type", "cover_type", "prem_rel"])
        return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
    TR.PRIOR_SUFFIX = "_p1v2"
    try:
        if l == "buildings":
            sc = np.asarray(TR.scratch(tr, va, l), float); st = glm(tr, va, None, TR.peril_own_override(tr, va, l, ["storm"])[0])
            return 0.9 * st + 0.1 * sc * st.sum() / sc.sum()
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        return glm(tr, va, ["gender"], ov) * TR.freq_correction(tr, va, l, ["works_from_home"])
    finally:
        TR.PRIOR_SUFFIX = ""


if sys.argv[2] != "riepilogo":
    i = int(sys.argv[2])
    em = M._prem(P.load(l)).assign(market=1.0)
    dv = TR.harmonize(pd.read_parquet(ROOT / f"data/model_s3/model_{l}.parquet"))
    dv = M._prem(dv[pd.util.hash_array(dv["policy_id"].to_numpy() + "_c") % 100 < 30].reset_index(drop=True)).assign(market=0.0)
    dv["row_id"] = "D" + dv["row_id"].astype(str)
    if "gender" not in dv:
        dv["gender"] = np.nan
    feats = sorted((set(c for c in M._features(dv, l) if c in em.columns) | {"gender", "prem_rel"}) - {"region"})
    cols = feats + ["market"]
    fname, tr, va = list(folds27(em))[i]
    y = va[P.TARGET].to_numpy(); preds = {"sub_21": sub21(tr, va)}
    pool = pd.concat([dv[cols + [P.TARGET, "exposure"]], tr[cols + [P.TARGET, "exposure"]]], ignore_index=True)
    Xp, cats = T.to_frame(pool, cols); Xv, _ = T.to_frame(va, cols, cats); Xt, _ = T.to_frame(tr, cols, cats)
    e = pool["exposure"].clip(lower=1e-3).to_numpy()
    for name, pp in PAR.items():
        prm = dict(T.GBM_PARAMS, num_threads=4, **{k: v for k, v in pp.items() if k != "rounds"})
        m = lgb.train(prm, lgb.Dataset(Xp, label=pool[P.TARGET].to_numpy() / e, weight=e), num_boost_round=pp["rounds"])
        pv = m.predict(Xv) * va["exposure"].to_numpy(); pt = m.predict(Xt) * tr["exposure"].to_numpy()
        preds[name] = pv * tr[P.TARGET].sum() / pt.sum()
    rows = [dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)) for k, p in preds.items()]
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(old):
        old = old[old.fold != fname]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)
    print(fname, {r["variante"]: round(r["forma"], 4) for r in rows}, flush=True)
else:
    r = pd.read_csv(OUT); b = r[r.variante == "sub_21"].set_index("fold")
    for c in ("forma", "own"):
        r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
    r["dir"] = r.fold.str.slice(0, 9); n = r.fold.nunique()
    s = r[r.variante != "sub_21"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
        vinte_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{n}"), vinte_own=("d_own", lambda x: f"{(x > 0).sum()}/{n}"))
    print(l, "Δ vs sub_21"); print(s.join(r[r.variante != "sub_21"].pivot_table(index="variante", columns="dir", values="d_forma").add_prefix("forma ")).round(1).to_string())
