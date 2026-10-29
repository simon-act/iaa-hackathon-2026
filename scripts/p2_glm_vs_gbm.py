"""Part 2 (27/10): where the developed GLM prior and GBM prior differ, and which one the emerging data support.
For each variable and class: GLM/GBM ratio (mean, weighted by predicted cost), A/E of the emerging target against each prior
(line level realigned), claims. 'Who wins' index = Σ_classes E·[(A/E_gbm − 1)² − (A/E_glm − 1)²] / Σ E: > 0 = the GLM
tracks the data better. Output: results_p2_llc/p2_glm_vs_gbm.csv. Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_glm_vs_gbm.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_resid_util import binned, DROP  # noqa: E402

rows = []
for l in P.LINES:
    df = P.load(l)
    g = pd.read_parquet(Path(P.RESULTS) / f"source_pred_{l}_glm.parquet").set_index("row_id").reindex(df.row_id)
    gb, gl = g["mu_src_gbm"].to_numpy(), g["mu_src"].to_numpy()
    y = df[P.TARGET].to_numpy(); gb = gb * y.sum() / gb.sum(); gl = gl * y.sum() / gl.sum()
    big = y > np.quantile(y, 0.999)
    cand = [c for c in M._features(df, l) if c in df.columns]
    for v in cand:
        t = pd.DataFrame({"g": binned(df[v]).to_numpy(), "y": y, "yx": np.where(big, 0, y), "gb": gb, "gl": gl,
                          "lr": np.log(gl / gb) * gb}).groupby("g").sum()
        t["ratio"] = np.exp(t.lr / t.gb); t["ae_gbm"] = t.y / t.gb; t["ae_glm"] = t.y / t.gl
        t["aex_gbm"] = t.yx / t.gb * (y.sum() / np.where(big, 0, y).sum()); t["aex_glm"] = t.yx / t.gl * (y.sum() / np.where(big, 0, y).sum())
        disp = float(np.sqrt((t.gb * np.log(t.ratio) ** 2).sum() / t.gb.sum()))
        win = float((t.gb * ((t.ae_gbm - 1) ** 2 - (t.ae_glm - 1) ** 2)).sum() / t.gb.sum())
        winx = float((t.gb * ((t.aex_gbm - 1) ** 2 - (t.aex_glm - 1) ** 2)).sum() / t.gb.sum())
        rows.append(dict(linea=l, var=v, livelli=len(t), diverg=disp, vince_glm=win, vince_glm_senza_top=winx,
                         dettaglio="; ".join(f"{i}: r{r.ratio:.2f} gbm{r.ae_gbm:.2f} glm{r.ae_glm:.2f}" for i, r in t.iterrows())))
r = pd.DataFrame(rows); r.to_csv(Path(P.RESULTS) / "p2_glm_vs_gbm.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
for l in P.LINES:
    print(f"\n=== {l}: variables where GLM and GBM diverge most (diverg = sd of the log ratio); vince_glm > 0 = the data side with the GLM")
    print(r[r.linea == l].sort_values("diverg", ascending=False).head(10)[["var", "diverg", "vince_glm", "vince_glm_senza_top", "dettaglio"]].round(4).to_string(index=False))
