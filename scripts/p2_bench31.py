"""Part 2 (28/10 evening, D tests, request: "try them all"): three practices not yet tested, vs p2_sub_21.
  tw05   : correction GLM with the exact offset for Tweedie p=1.5 (weight prior**0.5 instead of prior)        [motor, contents]
  fuoco1 : buildings, fire severity proportional to the sum insured (exposure rating, constant mean damage rate),
           fire level unchanged on train                                                                      [buildings]
  fuoco05: as fuoco1 but halfway (prior severity × (SI/SI_prior)^0.5 on log scale)                            [buildings]
  stag   : frequency correction (Poisson, offset = expected count from the prior) on start quarter and full year  [all]
Benches: '7' = 7 checks (F2024, F2023, K1–K5 by policy; rule ≥ 5/7 shape AND own) and '27' = strict (time direction ×
3 disjoint customer groups; rule: mean > 0 on shape and own, ≥ 4/6 on both, mean > 0 in both directions).
Rules fixed before the results. Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench31.py <line> <7|27> [fold,..]
Writes results_p2_llc/p2_bench31_<line>_<bench>_<fold>.csv (one per fold, to stay within the 3-minute limit)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
from p2_bench2 import folds as folds7  # noqa: E402
l, BANK = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None


def folds27(df):
    g = pd.util.hash_array(df["customer_id"].astype(str).to_numpy() + "_c27") % 3
    yr = df["year"].to_numpy()
    for a, b in ((2023, 2024), (2024, 2023)):
        for k in range(3):
            yield f"{a}-{b}c{k}", df[(yr == a) & (g != k)], df[(yr == b) & (g == k)]


def derive(df):
    df = df.copy()
    df["start_q"] = "q" + ((pd.to_datetime(df["period_start"]).dt.month - 1) // 3 + 1).astype(str)
    df["full_year"] = np.where(df["exposure"] >= 0.99, "intero", "parziale")
    return df


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def stag(tr, va):
    return TR.freq_correction(tr, va, l, ["start_q", "full_year"])


def motor(tr, va, wpow=1.0):
    TR.GLM_WPOW = wpow
    try:
        NG, FC = ["new_business", "gender", "prem_rel"], ["new_business", "gender", "fuel_type", "cover_type", "prem_rel"]
        a = lambda d: 0.5 * glm(tr, d, NG) + 0.5 * glm(tr, d, FC)
        return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
    finally:
        TR.GLM_WPOW = 1.0


def contents(tr, va, wpow=1.0):
    TR.GLM_WPOW = wpow; TR.PRIOR_SUFFIX = "_p1v2"
    try:
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        return glm(tr, va, ["gender"], ov) * TR.freq_correction(tr, va, l, ["works_from_home"])
    finally:
        TR.GLM_WPOW = 1.0; TR.PRIOR_SUFFIX = ""


def buildings(tr, va, fire_b=None):
    sc = np.asarray(TR.scratch(tr, va, l), float)
    TR.PRIOR_SUFFIX = "_p1v2"
    try:
        ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
        if fire_b is not None:
            parts = pd.concat([TR._prior_parts(tr, l), TR._prior_parts(va, l)])
            si = pd.concat([tr.set_index("row_id")["sum_insured"], va.set_index("row_id")["sum_insured"]]).reindex(parts.index).to_numpy(float)
            f, s, mu = parts["f_fire"].to_numpy(), parts["s_fire"].to_numpy(), parts["mu_src"].to_numpy()
            itr = np.arange(len(tr))
            # exposure-based severity: s_new = s × (SI / SI_impl)^b, with SI_impl = s / mean rate; rate chosen to keep
            # the expected fire cost on train unchanged (Σ f·s_new = Σ f·s)
            rate = np.sum(f[itr] * s[itr]) / np.sum(f[itr] * si[itr])
            s_full = rate * si                                   # b = 1: severity ∝ SI
            s_new = s * (s_full / s) ** fire_b
            s_new *= np.sum(f[itr] * s[itr]) / np.sum(f[itr] * s_new[itr])
            ov = ov.copy(); ov["mu_src"] = ov["mu_src"].to_numpy() * (1 + f * (s_new - s) / mu)
        b = glm(tr, va, None, ov)
    finally:
        TR.PRIOR_SUFFIX = ""
    return 0.9 * b + 0.1 * sc * b.sum() / sc.sum()


def variants(tr, va):
    if l == "motor":
        base = motor(tr, va); out = {"sub_21": base, "tw05": motor(tr, va, 0.5)}
    elif l == "contents":
        base = contents(tr, va); out = {"sub_21": base, "tw05": contents(tr, va, 0.5)}
    else:
        base = buildings(tr, va); out = {"sub_21": base, "fuoco1": buildings(tr, va, 1.0), "fuoco05": buildings(tr, va, 0.5)}
    out["stag"] = base * stag(tr, va)
    return out


df = derive(P.load(l)); df = M._prem(df) if l == "motor" else df
gen = folds7 if BANK == "7" else folds27
for fname, tr, va in gen(df):
    if ONLY and fname not in ONLY:
        continue
    y = va[P.TARGET].to_numpy(); rows = []
    for k, p in variants(tr, va).items():
        rows.append(dict(bank=BANK, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    pd.DataFrame(rows).to_csv(Path(P.RESULTS) / f"p2_bench31_{l}_{BANK}_{fname}.csv", index=False)
    print(fname, "ok", flush=True)
