"""Part 2 (30/10): does new business need a shape of its own in motor?
  nbx      : the sub_21 correction GLM + new-business-only terms (relative premium, gender, fuel and cover type inside new
             business; NaN / "ren" on renewals) = a new-business deviation from the common slope, shrunk by the ridge penalty
             already chosen on the holdout
  tw05     : D1 (exact Tweedie offset, weight prior^0.5)
  tw05_nbx : D1 + the new-business terms
Rule fixed BEFORE the results: candidate only if (1) it passes the strict bench 27 on all rows (mean > 0 on shape and own,
>= 4/6 on both, mean > 0 in both time directions) AND (2) the own judge on new-business rows only is > 0 in both directions.
Outcome: no variant passes (docs/MODEL_CARD.md); tw05_nbx was kept as a public-board probe only, not selected.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench33.py <7|27> [fold,..]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0], "motor"] + sys.argv[1:]
src = (Path(__file__).resolve().parent / "p2_bench31.py").read_text().split("df = derive(P.load(l))")[0]
exec(src)  # derive, glm, motor, folds27, folds7, P, TR, M
BANK = sys.argv[2]; ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None


def add_nb(df):
    df = df.copy(); nb = df["new_business"].astype(str).isin(["1", "True", "true", "yes", "Y"]) if df["new_business"].dtype != bool else df["new_business"]
    df["_isnb"] = nb.to_numpy()
    df["prem_rel_nb"] = np.where(nb, df["prem_rel"], np.nan)
    for c in ["gender", "fuel_type", "cover_type"]:
        df[c + "_nb"] = np.where(nb, df[c].astype(str), "ren")
    return df


def motor_nbx(tr, va, wpow=1.0):
    TR.GLM_WPOW = wpow
    try:
        X = ["prem_rel_nb", "gender_nb"]
        NG = ["new_business", "gender", "prem_rel"] + X
        FC = ["new_business", "gender", "fuel_type", "cover_type", "prem_rel"] + X + ["fuel_type_nb", "cover_type_nb"]
        a = lambda d: 0.5 * glm(tr, d, NG) + 0.5 * glm(tr, d, FC)
        return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
    finally:
        TR.GLM_WPOW = 1.0


df = add_nb(derive(M._prem(P.load(l))))
print("new-business share in train:", round(df["_isnb"].mean(), 3), flush=True)
gen = folds7 if BANK == "7" else folds27
for fname, tr, va in gen(df):
    if ONLY and fname not in ONLY:
        continue
    y = va[P.TARGET].to_numpy(); nb = va["_isnb"].to_numpy(); rows = []
    V = {"sub_21": motor(tr, va), "nbx": motor_nbx(tr, va), "tw05": motor(tr, va, 0.5), "tw05_nbx": motor_nbx(tr, va, 0.5)}
    for k, p in V.items():
        p = np.asarray(p, float)
        rows.append(dict(bank=BANK, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p),
                         own_nb=P.d2_own(va.row_id[nb], p[nb]), forma_nb=P.d2(y[nb], p[nb] * y[nb].sum() / p[nb].sum())))
    pd.DataFrame(rows).to_csv(Path(P.RESULTS) / f"p2_bench33_motor_{BANK}_{fname}.csv", index=False)
    print(fname, "ok", flush=True)
