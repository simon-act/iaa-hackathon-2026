"""Part 2 (28/10 evening): bench WEIGHTED to the 2025 mix (weights from p2_shift_weights.py) and bench on new business ONLY, to see whether the
bench/public-score disagreement is an effect of the portfolio shift. Validation: it must reproduce the direction of the comparisons with a known public outcome.
Variants by line (same functions as p2_bench27/p2_bench31): motor sub_21, sub_20 (premium on frequency), sub_18 (no premium),
tw05, stag; contents sub_21, current prior (= sub_19), tw05, stag; buildings sub_21, current prior (= sub_19), fuoco05, stag.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench32.py <line> <7|27>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
l, BANK = sys.argv[1], sys.argv[2]
ns = {"__name__": "b31", "__file__": str(ROOT / "scripts" / "p2_bench31.py")}; sys.argv = ["x", l, BANK]
exec(open(ROOT / "scripts" / "p2_bench31.py").read().split("df = derive(P.load(l))")[0], ns)
W = pd.read_parquet(Path(P.RESULTS) / f"p2_shift_weights_{l}.parquet").set_index("row_id")["w"]
J = pd.read_parquet(Path(P.RESULTS) / "judge_own.parquet").set_index("row_id")


def dev(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None); p = P.POWER
    return 2 * (np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def wd2(y, mu, w):
    mu = mu * np.sum(w * y) / np.sum(w * mu)
    return 1 - np.sum(w * dev(y, mu)) / np.sum(w * dev(y, np.full_like(y, np.sum(w * y) / np.sum(w), dtype=float)))


def scores(va, p):
    y = va[P.TARGET].to_numpy(float); w = W.reindex(va.row_id).to_numpy(); nb = va["new_business"].astype(bool).to_numpy()
    j = J.reindex(va.row_id); attr = np.clip(p - j["ll_expo"].to_numpy(), 1e-3, None) * j["pi_eff"].to_numpy(); yo = j["target_rep"].to_numpy()
    return dict(forma=wd2(y, p, np.ones_like(y)), own=wd2(yo, attr, np.ones_like(y)), forma_w=wd2(y, p, w), own_w=wd2(yo, attr, w),
                forma_nb=wd2(y[nb], p[nb], np.ones(nb.sum())))


def variants(tr, va):
    out = {}
    if l == "motor":
        out["sub_21"] = ns["motor"](tr, va); out["tw05"] = ns["motor"](tr, va, 0.5)
        NG, FC = ["new_business", "gender"], ["new_business", "gender", "fuel_type", "cover_type"]
        a = lambda d: 0.5 * ns["glm"](tr, d, NG) + 0.5 * ns["glm"](tr, d, FC)
        b = a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
        out["sub_18"] = b; out["sub_20"] = b * TR.freq_correction(tr, va, l, ["prem_rel"])
    elif l == "contents":
        out["sub_21"] = ns["contents"](tr, va); out["tw05"] = ns["contents"](tr, va, 0.5)
        TR.PRIOR_SUFFIX = ""
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        out["sub_19"] = ns["glm"](tr, va, ["gender"], ov) * TR.freq_correction(tr, va, l, ["works_from_home"])
    else:
        out["sub_21"] = ns["buildings"](tr, va); out["fuoco05"] = ns["buildings"](tr, va, 0.5)
        sc = np.asarray(TR.scratch(tr, va, l), float)
        st0 = ns["glm"](tr, va, None, TR.peril_own_override(tr, va, l, ["storm"])[0])
        out["sub_19"] = 0.9 * st0 + 0.1 * sc * st0.sum() / sc.sum()
    out["stag"] = out["sub_21"] * ns["stag"](tr, va)
    return out


df = ns["derive"](P.load(l)); df = M._prem(df) if l == "motor" else df
gen = ns["folds7"] if BANK == "7" else ns["folds27"]
rows = []
for fname, tr, va in gen(df):
    for k, p in variants(tr, va).items():
        rows.append(dict(linea=l, bank=BANK, fold=fname, variante=k, **scores(va, np.asarray(p, float))))
pd.DataFrame(rows).to_csv(Path(P.RESULTS) / f"p2_bench32_{l}_{BANK}.csv", index=False)
print(l, BANK, "ok")
