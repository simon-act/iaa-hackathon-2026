"""Part 2 (25/10): 7-check bench (same folds as p2_bench.py) for new business by channel.
Rationale: A/E new business / renewals relative to the prior = 1.20 in broker (CI90 1.00–1.45) and 0.97 in direct (0.80–1.16);
in the test, new business is 53% broker, 19% direct, 4% aggregators. Variants (correction GLM with offset = prior):
gender+nb (= sub_07), gender+nb broker only, gender+nb×channel, gender+channel (control).
Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench2.py [lines] → results_p2/p2_bench2.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402

VARS = {"genere+nb": ["gender", "new_business"], "genere+nb_broker": ["gender", "nb_broker"],
        "genere+nb×canale": ["gender", "nb_ch"], "genere+canale": ["gender", "channel"]}


def derive(df):
    df = df.copy()
    nb = df["new_business"].astype(bool)
    df["nb_broker"] = nb & (df["channel"] == "broker")
    df["nb_ch"] = np.where(nb, "nb_" + df["channel"].astype(str), "rinnovo")
    return df


def folds(df):
    yield "F2024", df[df.year == 2023], df[df.year == 2024]
    yield "F2023", df[df.year == 2024], df[df.year == 2023]
    g = pd.util.hash_array((df["policy_id"] + "_k").to_numpy()) % 5
    for k in range(5):
        yield f"K{k + 1}", df[g != k], df[g == k]


if __name__ == "__main__":
    lines = sys.argv[1].split(",") if len(sys.argv) > 1 else ["motor", "contents"]
    rows, coefs = [], []
    for l in lines:
        df = derive(P.load(l))
        for fname, tr, va in folds(df):
            y = va[P.TARGET].to_numpy()
            preds = {"livello": TR.src_level(tr, va, l)}
            for k, v in VARS.items():
                TR.CORRECTION_VARS, old = v, TR.CORRECTION_VARS
                try:
                    preds[k] = TR.src_glm(tr, va, l); coefs.append(dict(linea=l, fold=fname, variante=k, **TR.src_glm.last))
                finally:
                    TR.CORRECTION_VARS = old
            for k, p in preds.items():
                rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                                 own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
            print(l, fname, "done", flush=True)
    r = pd.DataFrame(rows)
    b = r[r.variante == "genere+nb"].set_index(["linea", "fold"])
    for c in ("forma", "own", "unc"):
        r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
    r.to_csv(Path(P.RESULTS) / "p2_bench2.csv", index=False)
    pd.DataFrame(coefs).to_csv(Path(P.RESULTS) / "p2_bench2_coef.csv", index=False)
    pd.set_option("display.width", 220)
    s = r[r.variante != "genere+nb"].groupby(["linea", "variante"]).agg(
        forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
        pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"),
        pos_unc=("d_unc", lambda x: f"{(x > 0).sum()}/{len(x)}"))
    print("Δ e·4 vs gender+nb (sub_07)"); print(s.round(1).to_string())
    print(r.pivot_table(index=["linea", "variante"], columns="fold", values="d_forma").round(1).to_string())
