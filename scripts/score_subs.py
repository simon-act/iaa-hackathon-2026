"""Internal score (OOF, folds 2023/2024, Kaggle weights) of each uploaded submission, next to the Kaggle public score (19/10).
For submissions that are transformations (level, power) the transformation is applied to the base model's OOF.
Judge for each = target_b of the configuration it was built with (the one used at the time to decide).
Writes results_s3/subs_internal_vs_lb.csv and results_s3/all_models_scores.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P  # noqa: E402

W = {"motor": 254986, "buildings": 184155, "contents": 187498}
COLS = ["row_id", "line", "fold", "target_b", "pred"]


def rd(cfg, name):
    o = pd.read_parquet(ROOT / cfg / "oof" / f"{name}.parquet", columns=COLS)
    o["pred"] = o["pred"].astype("float64")
    return o


def score(o, pred=None):
    p = o["pred"].to_numpy() if pred is None else pred
    by = {}
    for (l, f), idx in o.groupby(["line", "fold"]).indices.items():
        by[(l, f)] = P.d2(o["target_b"].to_numpy()[idx], p[idx])
    return sum(by[(l, f)] * W[l] for l in W for f in ("F2023", "F2024")) / (sum(W.values()) * 2)


def by_line(o, mult):
    """mult: dict line -> function(pred array) -> pred array (per-line transformation)."""
    p = o["pred"].to_numpy().copy()
    for l, fn in mult.items():
        for f in ("F2023", "F2024"):
            m = ((o["line"] == l) & (o["fold"] == f)).to_numpy()
            p[m] = fn(p[m], m)
    return p


def pw(b):
    return lambda p, m: (p ** b) * p.sum() / (p ** b).sum()


def lv(c):
    return lambda p, m: p * c


def mix(o, parts, weights):
    """per-line blend from the OOF of components aligned on row_id."""
    p = o["pred"].to_numpy().copy()
    for l, w in weights.items():
        m = (o["line"] == l).to_numpy()
        p[m] = sum(wi * pi[m] for wi, pi in zip(w, parts))
    return p


def main():
    S = {}
    # --- original configuration (results/) ---
    g = rd("results", "glm_b"); b = rd("results", "gbm_b")
    S["sub_01"] = score(g, 0.5 * g["pred"].to_numpy() + 0.5 * b["pred"].to_numpy())
    S["sub_03"] = score(rd("results", "fs_peril_r4"))
    s5 = rd("results", "sub05").sort_values("row_id").reset_index(drop=True)
    S["sub_05"] = score(s5)
    S["sub_06"] = score(s5, by_line(s5, {"buildings": pw(1.05)}))
    fa = rd("results", "fs2_all_ll_cred").set_index("row_id").reindex(s5["row_id"])["pred"].to_numpy()
    q7 = rd("results", "qwen_gbm73_cred").set_index("row_id").reindex(s5["row_id"])["pred"].to_numpy()
    p7 = s5["pred"].to_numpy().copy(); mb = (s5["line"] == "buildings").to_numpy()
    p7[mb] = 0.55 * fa[mb] + 0.45 * q7[mb]
    s7 = s5.assign(pred=p7)
    S["sub_07"] = score(s7)
    mod = pd.read_parquet(ROOT / "data/model/model_buildings.parquet", columns=["row_id", "year", "period_end"]).set_index("row_id")
    pe = pd.to_datetime(mod["period_end"].reindex(s7["row_id"]).to_numpy())
    early = mb & ~((pe.month == 12) & (pe.day >= 30))
    p8 = p7.copy(); p8[early] *= 0.85
    for f in ("F2023", "F2024"):   # line level fixed as in the probe
        m = mb & (s7["fold"] == f).to_numpy(); p8[m] *= p7[m].sum() / p8[m].sum()
    S["sub_08"] = score(s7, p8)
    S["sub_09"] = S["sub_07"]          # same model, final fit on all policies
    S["sub_10"] = score(s7, by_line(s7, {"motor": lv(1.04)}))
    S["sub_11"] = score(s7, by_line(s7, {"buildings": lv(1.04)}))
    S["sub_12"] = score(s7, by_line(s7, {"contents": lv(1.04)}))
    S["sub_13"] = score(s7, by_line(s7, {"motor": lv(0.976)}))
    S["sub_15"] = score(rd("results", "blend_opt"))
    S["sub_16"] = score(rd("results_tro", "blend_opt"))
    S["sub_17"] = score(rd("results_v3", "blend_opt"))
    s3 = rd("results_s3", "blend_opt")
    S["sub_18"] = score(s3)
    S["sub_19"] = score(s3, by_line(s3, {"motor": lv(581.9 / 568.9)}))   # Pareto: same model, different motor LL load
    S["sub_20"] = score(s3, by_line(s3, {"buildings": lv(1.02)}))
    S["sub_21"] = score(s3, by_line(s3, {"motor": pw(0.95), "buildings": pw(1.05), "contents": pw(0.95)}))
    a4 = rd("results_s4", "fs2_v2_cred").set_index("row_id").reindex(s3["row_id"])["pred"].to_numpy()
    qb = rd("results_s3", "qwen_gbm73_cred").set_index("row_id").reindex(s3["row_id"])["pred"].to_numpy()
    wts = json.loads((ROOT / "results_s3/blend_weights.json").read_text())["weights"]
    S["sub_23"] = score(s3, mix(s3, [a4, qb], {l: [wts[l]["fs2_v2_cred"], wts[l]["qwen_gbm73_cred"]] for l in wts}))
    S["sub_24"] = score(s3, by_line(s3, {"buildings": pw(1.08) if False else (lambda p, m: (p ** 1.08) * p.sum() / (p ** 1.08).sum() * 1.02)}))
    pd.Series(S).round(5).rename("interno").to_csv(ROOT / "results_s3/subs_internal.csv")
    print(pd.Series(S).round(5))


if __name__ == "__main__":
    main()
