"""FROZEN JUDGE for the Part 2 (emerging market) loop (25/10). Do not modify.

Task: predict the 2025 claim cost of emerging-market policy rows. A prior from the developed market is given per row
(`prior_mu_src` = expected cost; `prior_f_<peril>` / `prior_s_<peril>` = expected claim count and severity per peril,
`prior_c_src` = the prior's calibration factor). The emerging data (2 years, ~110k rows) should set the level and only
the corrections they can support.

Seven checks per line: F2024 (fit 2023 -> score 2024), F2023 (fit 2024 -> score 2023), K1..K5 (five groups of policies
across both years). Reference = prior x level (level = sum(target_b) / sum(prior) on train).
Gains over the reference, Tweedie p = 1.5, level of each prediction reset to the actual total (shape only):
  own  = D2 on reported developed cost (target_rep) vs (pred - ll_expo) * pi_eff  (no IBNR spreading)
  forma = D2 on target_b (official-like)
  unc  = like own but with the true large losses added to the actual (reported)
Lines weighted by 2025 test rows. pooled_mean = min(mean own gain, mean forma gain) over the 7 checks: a change must
help both. score = mean own gain. Usage: python -I eval_experiment.py <experiment.py> [--columns-out columns.json]
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = {"motor": 90521, "buildings": 52295, "contents": 66247}
HIDE = ["target_b", "target_rep", "pi_eff", "ll_expo", "excess_onl"]


def d2(y, mu, p=1.5):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)

    def dev(m):
        return 2 * (np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p)) - y * np.power(m, 1 - p) / (1 - p)
                    + np.power(m, 2 - p) / (2 - p))
    return 1 - dev(mu).sum() / dev(np.full_like(y, y.mean())).sum()


def judges(va, pred):
    y = va["target_b"].to_numpy()
    forma = d2(y, pred * y.sum() / pred.sum())
    e = np.clip(pred - va["ll_expo"].to_numpy(), 1e-3, None) * va["pi_eff"].to_numpy()
    yr = va["target_rep"].to_numpy()
    own = d2(yr, e * yr.sum() / e.sum())
    yu = yr + va["excess_onl"].fillna(0).to_numpy()
    unc = d2(yu, (pred * va["pi_eff"].to_numpy()) * yu.sum() / (pred * va["pi_eff"].to_numpy()).sum())
    return own, forma, unc


def checks(d):
    yield "F2024", d[d.year == 2023], d[d.year == 2024]
    yield "F2023", d[d.year == 2024], d[d.year == 2023]
    g = pd.util.hash_array((d["policy_id"] + "_k").to_numpy()) % 5
    for k in range(5):
        yield f"K{k + 1}", d[g != k], d[g == k]


def main():
    path = sys.argv[1]
    cols_out = sys.argv[sys.argv.index("--columns-out") + 1] if "--columns-out" in sys.argv else None
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("experiment_under_test", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = pd.read_parquet(HERE / "data_p2.parquet")
    gains, columns = {}, {}
    for line in W:
        d = data[data["line"] == line].dropna(axis=1, how="all").reset_index(drop=True)
        keep = [c for c in d.columns if c not in HIDE and c not in ("line",)]
        columns[line] = {c: str(d[c].dtype) for c in keep}
        for name, tr, va in checks(d):
            tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
            pred = np.asarray(mod.fit_predict(tr[keep + HIDE].copy(), va[keep].copy(), line), dtype=float)
            if pred.shape != (len(va),) or not np.isfinite(pred).all() or (pred < 0).any():
                raise RuntimeError("predictions must be finite, >= 0, one per valid row")
            ref = va["prior_mu_src"].to_numpy() * tr["target_b"].sum() / tr["prior_mu_src"].sum()
            a, b = judges(va, pred), judges(va, ref)
            gains[f"{line}|{name}"] = [round(float(x - y), 6) for x, y in zip(a, b)]
    tot = sum(W.values())
    mean = lambda j: sum(np.mean([gains[f"{l}|{c}"][j] for c in ("F2024", "F2023", "K1", "K2", "K3", "K4", "K5")]) * W[l] for l in W) / tot
    own, forma, unc = mean(0), mean(1), mean(2)
    res = {"by": {k: v[0] for k, v in gains.items()}, "own": round(own, 6), "forma": round(forma, 6), "unc": round(unc, 6),
           "pooled_mean": round(min(own, forma), 6), "score": round(own, 6), "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
