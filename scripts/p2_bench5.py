"""Part 2 (25/10): "more credibility to the new data", in two structured ways, on the 7-check bench (folds from p2_bench2).
Rationale: emerging / prior ratios by peril differ widely (buildings: water 0.59, fire 1.32, storm 0.98; motor:
glass 0.73, theft 0.94), while sub_07 uses a single level factor per line.
1. Peril level (frequency and severity) with credibility toward the line: k = claim count at which the peril gets 50% weight;
   k (freq, sev) = (25, 50) more credibility, (100, 200), (400, 800) less. Then the same sub_07 corrections (gender, nb) on top.
2. Blend with the emerging-only model (scratch): w = 0.1 / 0.2 / 0.3 / 0.5 on top of sub_07 (buildings: already 0.1 in sub_07).
Rule: adopted if it wins ≥ 5/7 checks on shape and own. Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench5.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

VARS = {"motor": ["gender", "new_business"], "contents": ["gender"], "buildings": None}
KS = [(25, 50), (100, 200), (400, 800)]


def corrected(tr, va, l, override=None):
    TR._PRIOR_OVERRIDE = override
    try:
        if VARS[l]:
            TR.CORRECTION_VARS = VARS[l]
            return np.asarray(TR.src_glm(tr, va, l), float)
        return np.asarray(TR.src_level(tr, va, l), float)
    finally:
        TR._PRIOR_OVERRIDE = None


rows, logs = [], []
for l in P.LINES:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        base = corrected(tr, va, l)
        sc = np.asarray(TR.scratch(tr, va, l), float); sc = sc * base.sum() / sc.sum()
        w0 = 0.1 if l == "buildings" else 0.0
        preds = {"sub_07": (1 - w0) * base + w0 * sc}
        for kf, ks in KS:
            p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=kf, k_sev=ks), float)
            p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=kf, k_sev=ks), float)
            logs.append(dict(linea=l, fold=fname, k=f"{kf}/{ks}", **{p: v["a"] * v["b"] for p, v in TR.src_peril.last_log["perils"].items()}))
            ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
            b2 = corrected(tr, va, l, ov)
            preds[f"perilo k={kf}/{ks}"] = (1 - w0) * b2 + w0 * sc * b2.sum() / sc.sum()
        for w in (0.1, 0.2, 0.3, 0.5):
            if w != w0:
                preds[f"scratch w={w}"] = (1 - w) * base + w * sc
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "done", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench5.csv", index=False); pd.DataFrame(logs).to_csv(Path(P.RESULTS) / "p2_bench5_livelli.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_07"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_07"); print(s.round(1).to_string())
print(r.pivot_table(index=["linea", "variante"], columns="fold", values="d_forma").round(1).to_string())
