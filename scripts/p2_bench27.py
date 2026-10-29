"""Part 2 (28/10, A2): more honest bench. The K folds of the 7-check bench split by policy but share the years (same year
in train and valid): the 7 checks are worth perhaps 3–4. Here: time direction only (train on one year, valid on the other) × 3 disjoint
CUSTOMER groups (no customer in both train and valid) = 6 checks.
Rule fixed before the results: adopted if mean > 0 on shape AND own, ≥ 4/6 on both, and mean > 0 in both time directions.
Variants = components of p2_sub_21 removed one at a time, and sub_20. Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench27.py <line> [variants,..]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
l = sys.argv[1]; ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def folds27(df):
    g = pd.util.hash_array(df["customer_id"].astype(str).to_numpy() + "_c27") % 3
    yr = df["year"].to_numpy()
    for a, b in ((2023, 2024), (2024, 2023)):
        for k in range(3):
            yield f"{a}→{b} c{k}", df[(yr == a) & (g != k)], df[(yr == b) & (g == k)]


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def variants(tr, va):
    out = {}
    if l == "motor":
        NG, FC = ["new_business", "gender"], ["new_business", "gender", "fuel_type", "cover_type"]
        def run(extra=(), half=True, gbm=True, freqprem=False):
            ex = list(extra)
            a = (lambda d: 0.5 * glm(tr, d, NG + ex) + 0.5 * glm(tr, d, FC + ex)) if half else (lambda d: glm(tr, d, NG + ex))
            p = a(va)
            if gbm:
                p = p * TR.gbm_freq_correction(tr, va, l, a(tr))
            if freqprem:
                p = p * TR.freq_correction(tr, va, l, ["prem_rel"])
            return p
        todo = {"sub_21": lambda: run(["prem_rel"]), "sub_20 (premio sulla frequenza)": lambda: run(freqprem=True),
                "senza premio (= sub_18)": lambda: run(), "senza GBM numero sinistri": lambda: run(["prem_rel"], gbm=False),
                "senza fuel/cover": lambda: run(["prem_rel"], half=False)}
        for k, f in todo.items():
            if k == "sub_21" or ONLY is None or k in ONLY:
                out[k] = f()
    elif l == "buildings":
        sc = np.asarray(TR.scratch(tr, va, l), float); mix = lambda p, w=0.1: (1 - w) * p + w * sc * p.sum() / sc.sum()
        TR.PRIOR_SUFFIX = "_p1v2"; st = glm(tr, va, None, TR.peril_own_override(tr, va, l, ["storm"])[0]); lv = glm(tr, va, None)
        TR.PRIOR_SUFFIX = ""; st0 = glm(tr, va, None, TR.peril_own_override(tr, va, l, ["storm"])[0])
        out.update({"sub_21": mix(st), "senza storm": mix(lv), "senza 10% solo emerging": st, "prior attuale invece di P1 v2": mix(st0)})
    else:
        def cont(suf, gender=True, wfh=True, peril=True):
            TR.PRIOR_SUFFIX = suf
            try:
                ov = None
                if peril:
                    p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
                    p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
                    ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
                p = glm(tr, va, ["gender"] if gender else None, ov)
                return p * TR.freq_correction(tr, va, l, ["works_from_home"]) if wfh else p
            finally:
                TR.PRIOR_SUFFIX = ""
        out.update({"sub_21": cont("_p1v2"), "senza genere": cont("_p1v2", gender=False), "senza lavoro da casa": cont("_p1v2", wfh=False),
                    "senza livelli per perilo": cont("_p1v2", peril=False), "prior attuale invece di P1 v2": cont("")})
    return out


rows = []
df = P.load(l); df = M._prem(df) if l == "motor" else df
for fname, tr, va in folds27(df):
    y = va[P.TARGET].to_numpy()
    for k, p in variants(tr, va).items():
        rows.append(dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_21"].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r["dir"] = r.fold.str.slice(0, 9)
tag = "" if ONLY is None else "_" + str(abs(hash(",".join(ONLY))) % 1000)
r.to_csv(Path(P.RESULTS) / f"p2_bench27_{l}{tag}.csv", index=False)
print(l, "(Δ of the variant vs sub_21: negative = sub_21 better, i.e. the removed component is useful)")
s = r[r.variante != "sub_21"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
    vinte_forma=("d_forma", lambda x: f"{(x > 0).sum()}/6"), vinte_own=("d_own", lambda x: f"{(x > 0).sum()}/6"))
s = s.join(r[r.variante != "sub_21"].pivot_table(index="variante", columns="dir", values="d_forma").add_prefix("forma "))
print(s.round(1).to_string())
