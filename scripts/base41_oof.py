"""OOF (folds 2023/2024) of the sub_41 blend, from the saved components (21/10 evening). Approximations: motor GBM without the gross target
(qwen_gbm73fix_cred). Buildings with credibility excluding subsidence, as in the final run. Output results_s3/oof/base41.parquet"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS
part = lambda n, l, f: pd.read_parquet(R / f"oof_parts/{n}_{l}_{f}.parquet").set_index("row_id")["pred"]
ref = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "line", "fold", "year", "exposure", "target_b"])
QB = pd.read_parquet(R / "oof/qwen_gbm73_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]
SPEC = {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_cred", 0.9, False), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True),
        "contents": ("fs2_v2_sevmix_cred", None, 0.8, False)}
out = []
for l, (a, b, w, ex) in SPEC.items():
    for f in P.FOLDS:
        s = ref[(ref.line == l) & (ref.fold == f)].copy()
        pa = part(a, l, f).reindex(s.row_id).to_numpy(); pb = (part(b, l, f) if b else QB).reindex(s.row_id).to_numpy()
        if ex:
            pa = X.apply_excl(X.uncred(pa, s, l, 8), s, l, 8); pb = X.apply_excl(X.uncred(pb, s, l, 8), s, l, 8)
        s["pred"] = w * pa + (1 - w) * pb; out.append(s)
o = pd.concat(out, ignore_index=True); o.to_parquet(R / "oof/base41.parquet", index=False)
W = {"motor": 254986, "buildings": 184155, "contents": 187498}
for f in P.FOLDS:
    z = o[o.fold == f]
    print(f, "official", round(sum(P.d2(z[z.line == l].target_b, z[z.line == l].pred) * W[l] for l in W) / sum(W.values()), 5),
          "own", round(sum(P.d2_own(z[z.line == l].row_id, z[z.line == l].pred.to_numpy()) * W[l] for l in W) / sum(W.values()), 5))
