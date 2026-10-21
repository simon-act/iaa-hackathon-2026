"""FROZEN JUDGE for the history (experience rating) loop (21/10). Do not modify.

Task: improve how the CURRENT BEST MODEL (blend submitted as sub_41) uses each policy's claims history.
Data: out-of-fold predictions of the best model on 2023 and 2024 (`base_pred`, already with experience rating k=8;
`base_nocred` = the same without experience rating), history columns from storia.parquet, allowed feature columns.

Temporal cross-fit: fit on 2023 rows -> predict 2024 rows, and fit on 2024 -> predict 2023.
Judge = "own" D2 (no IBNR spreading): actual = reported developed cost `target_rep`;
expected = (prediction - ll_expo) * pi_eff, rescaled to the actual total of the line and year.
Gain = D2_own(new) - D2_own(base_pred), Tweedie p=1.5, lines weighted like Kaggle.
The loop optimises pooled_mean = min(gain_2023, gain_2024). The official D2 (target_b) is reported too.
The total predicted level of each line and year is forced equal to base_pred (shape only).
Usage: python -I eval_experiment.py <experiment.py> [--columns-out columns.json]
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import prepare as P  # noqa: E402

W = {"motor": 254986, "buildings": 184155, "contents": 187498}
HIDE = ["target_b", "target_rep", "pi_eff", "ll_expo"]


def d2(y, mu, p=1.5):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)

    def dev(m):
        return 2 * (np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p)) - y * np.power(m, 1 - p) / (1 - p)
                    + np.power(m, 2 - p) / (2 - p))
    return 1 - dev(mu).sum() / dev(np.full_like(y, y.mean())).sum()


def d2_own(df, pred):
    e = np.clip(pred - df["ll_expo"].to_numpy(), 1e-3, None) * df["pi_eff"].to_numpy()
    y = df["target_rep"].to_numpy()
    return d2(y, e * y.sum() / e.sum())


def main():
    path = sys.argv[1]
    cols_out = sys.argv[sys.argv.index("--columns-out") + 1] if "--columns-out" in sys.argv else None
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("experiment_under_test", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = pd.read_parquet(HERE / "base_oof.parquet")
    hist = pd.read_parquet(HERE / "storia.parquet")
    by, byu, columns = {}, {}, {}
    for line in ["motor", "buildings", "contents"]:
        feat = P.load(line)
        feats = [c for c in P.allowed_features(feat) if c not in ("exposure",)]
        b = base[base["line"] == line].drop(columns=["exposure"]).merge(feat[["row_id", "exposure"] + feats], on="row_id")
        hcols = [c for c in hist.columns if c != "row_id" and c not in b.columns and hist[hist.row_id.isin(b.row_id)][c].notna().any()]
        b = b.merge(hist[["row_id"] + hcols], on="row_id", how="left")
        keep = feats + hcols + ["exposure", "year", "base_pred", "base_nocred"]
        columns[line] = {c: str(b[c].dtype) for c in keep}
        for fit_fold, score_fold in (("F2023", "F2024"), ("F2024", "F2023")):
            tr = b[b["fold"] == fit_fold].reset_index(drop=True)
            va = b[b["fold"] == score_fold].reset_index(drop=True)
            pred = np.asarray(mod.fit_predict(tr[keep + HIDE].copy(), va[keep].copy(), line), dtype=float)
            if pred.shape != (len(va),):
                raise RuntimeError(f"wrong prediction shape {pred.shape} for {len(va)} rows")
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise RuntimeError("predictions must be finite and >= 0")
            pred = pred * va["base_pred"].sum() / pred.sum()
            by[f"{line}|{score_fold}"] = round(float(d2_own(va, pred) - d2_own(va, va["base_pred"].to_numpy())), 6)
            byu[f"{line}|{score_fold}"] = round(float(d2(va["target_b"], pred) - d2(va["target_b"], va["base_pred"])), 6)
    tot = sum(W.values())
    g = {f: sum(by[f"{l}|{f}"] * W[l] for l in W) / tot for f in ("F2023", "F2024")}
    gu = {f: sum(byu[f"{l}|{f}"] * W[l] for l in W) / tot for f in ("F2023", "F2024")}
    res = {"by": by, "pooled": {k: round(v, 6) for k, v in g.items()}, "official": {k: round(v, 6) for k, v in gu.items()},
           "pooled_mean": round(min(g.values()), 6), "score": round(float(np.mean(list(g.values()))), 6),
           "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
