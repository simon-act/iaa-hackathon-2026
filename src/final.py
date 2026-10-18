"""Controllo sull'holdout chiuso e submission finale da una combinazione di modelli.

Uso:
  python -I final.py holdout              # D² sull'holdout (stima 2021-23 senza holdout → holdout 2024)
  python -I final.py submit <out.csv> [--all]   # stima su 2021-24 (senza holdout; --all = tutte le polizze) → previsione test 2025
I modelli e i pesi per linea vengono da results/blend_weights.json (scritto da blend.py).
Confronto di riferimento sull'holdout: sub_01 = 0,5·glm_b + 0,5·gbm_b.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
import train as T  # noqa: E402
import models_extra as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = P.RESULTS


def fn(name):
    return getattr(T, name) if hasattr(T, name) else getattr(M, name)


def main():
    mode = sys.argv[1]
    T.NUM_ROUNDS.update(json.loads(T.STATE.read_text()).get("num_rounds", {}))
    bw = json.loads((RES / "blend_weights.json").read_text())
    W = bw["weights"]
    models = sorted({m for w in W.values() for m, v in w.items() if v > 0} | ({"glm_b", "gbm_b"} if mode == "holdout" else set()))
    t0 = time.time()
    if mode == "holdout":
        rows, ys, preds = [], [], {}
        for line in P.LINES:
            df = P.load(line)
            ho = P.is_holdout(df["policy_id"])
            tr = df[(~ho) & df["year"].isin([2021, 2022, 2023])]
            va = df[ho & (df["year"] == 2024)]
            y = va[P.TARGET].to_numpy()
            pl = {m: np.asarray(fn(m)(tr, va, line), float) for m in models}
            blend = sum(W[line].get(m, 0) * pl[m] for m in pl)
            sub01 = 0.5 * pl["glm_b"] + 0.5 * pl["gbm_b"]
            r = {"line": line, "rows": len(va), "new_blend": P.d2(y, blend), "sub01_blend": P.d2(y, sub01)}
            r.update({m: P.d2(y, p) for m, p in pl.items()})
            rows.append(r); ys.append(y)
            preds.setdefault("new_blend", []).append(blend); preds.setdefault("sub01_blend", []).append(sub01)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
        y = np.concatenate(ys)
        pooled = {k: round(P.d2(y, np.concatenate(v)), 5) for k, v in preds.items()}
        out = {"by_line": rows, "pooled": pooled, "weights": W}
        (RES / "holdout_check.json").write_text(json.dumps(out, indent=1, default=float))
        print("POOLED holdout:", pooled)
    elif mode == "submit":
        out_csv = sys.argv[2]
        use_all = "--all" in sys.argv   # stima finale su TUTTE le polizze (holdout compreso): +10% dati, stessa struttura
        subs = []
        for line in P.LINES:
            df = P.load(line)
            if not use_all:
                df = df[~P.is_holdout(df["policy_id"])]
            te = P.load(line, "test")
            pl = {}
            for m in models:
                if W[line].get(m, 0) > 0:
                    pl[m] = np.asarray(fn(m)(df, te, line), float)
                    print(f"  {line} {m}: rate {pl[m].sum() / te['exposure'].sum():.1f}", f"{time.time() - t0:.0f}s", flush=True)
            p = sum(W[line][m] * pl[m] for m in pl)
            d = pd.DataFrame({"row_id": te["row_id"].to_numpy(), "prediction": p, "line": line,
                              "exposure": te["exposure"].to_numpy()})
            for m in pl:
                d[f"p_{m}"] = pl[m]
            subs.append(d)
        s = pd.concat(subs, ignore_index=True)
        s.to_parquet(RES / f"pred_{Path(out_csv).stem}.parquet", index=False)
        ss = pd.read_parquet(Path(P.CFG["raw"]) / "sample_submission.parquet")
        assert len(s) == len(ss) and set(s["row_id"]) == set(ss["row_id"]), "row_id diversi dalla sample"
        assert np.isfinite(s["prediction"]).all() and (s["prediction"] >= 0).all()
        s = ss[["row_id"]].merge(s, on="row_id", how="left")
        s["prediction"] = s["prediction"].round(3)
        s[["row_id", "prediction"]].to_csv(out_csv, index=False)
        print(s.groupby("line").apply(lambda g: g["prediction"].sum() / g["exposure"].sum(), include_groups=False))


if __name__ == "__main__":
    main()
