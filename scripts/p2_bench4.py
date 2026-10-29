"""Part 2 (25/10): Part 1 experience rating (credibility on the policy's claims history, src/experience.py) on top of the
sub_07 model. Never tested in Part 2. k fixed from Part 1, not estimated here: motor 16, buildings and contents 8.
k grid only as a diagnostic. Folds from p2_bench2 (F2023 not informative: 2023 rows have no history → M = 1).
Rule: adopted if it wins ≥ 5 of the 6 informative checks on shape and own.
Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench4.py → results_p2/p2_bench4.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, experience as X  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

K_P1 = {"motor": 16, "buildings": 8, "contents": 8}
VARS = {"motor": ["gender", "new_business"], "contents": ["gender"]}


def base_pred(tr, va, l):
    """sub_07 model by line: motor gender+nb, contents gender, buildings 0.9 prior×level + 0.1 scratch."""
    if l in VARS:
        TR.CORRECTION_VARS = VARS[l]
        return np.asarray(TR.src_glm(tr, va, l), float)
    b = np.asarray(TR.src_level(tr, va, l), float); sc = np.asarray(TR.scratch(tr, va, l), float)
    return 0.9 * b + 0.1 * sc * b.sum() / sc.sum()


rows = []
for l in P.LINES:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        b = base_pred(tr, va, l)
        preds = {"sub_07": b}
        for k in sorted({K_P1[l], 2, 4, 8, 16, 32}):
            preds[f"cred_k{k}" + (" (Parte 1)" if k == K_P1[l] else "")] = X.apply(b, va, l, k)
        for name, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=name, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "done", flush=True)
r = pd.DataFrame(rows)
bb = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - bb.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench4.csv", index=False)
inf = r[(r.variante != "sub_07") & (r.fold != "F2023")]
s = inf.groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
pd.set_option("display.width", 220)
print("Δ e·4 vs sub_07, 6 informative checks"); print(s.round(1).to_string())
print(r[r.variante.str.contains("Parte 1")].pivot_table(index="linea", columns="fold", values="d_forma").round(1).to_string())
