"""23/10: final candidates on the frozen holdout (2024 rows of the 10% of policies never used), per-line blend as in
build_final. Judges: official and own (no spreading). Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/holdout_judge.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS / "holdout_parts"
def part(n, l):
    return pd.read_parquet(R / f"{n}_{l}.parquet")
CAND = {"sub_36≈": {"motor": ("fs2_v2f_cred", "qwen_gbm73fix_gross_cred", 0.9, False, 8), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True, 8), "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False, 8)},
        "sub_40":  {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_gross_cred", 0.9, False, 8), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True, 8), "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False, 8)},
        "sub_47 (k16 auto)": {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_gross_cred", 0.9, False, 16), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True, 8), "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False, 8)},
        "sub_48/50 (senza alluvione)": {"motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_gross_cred", 0.9, False, 8), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True, 8), "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False, 8)}}
W = {l: len(P.load(l, "test")) for l in P.LINES}
rows = []
for name, spec in CAND.items():
    r = {"cand": name}; tot_u = tot_o = 0
    for l, (a, b, w, ex, k) in spec.items():
        A = part(a, l).set_index("row_id"); B = part(b, l).set_index("row_id").reindex(A.index)
        s = A.reset_index()[["row_id", "exposure"]]; pa, pb = A.pred.to_numpy(), B.pred.to_numpy()
        if ex:
            pa, pb = X.apply_excl(X.uncred(pa, s, l, 8), s, l, 8), X.apply_excl(X.uncred(pb, s, l, 8), s, l, 8)
        p = w * pa + (1 - w) * pb
        if k != 8:
            p = X.apply(w * X.uncred(pa, s, l, 8) + (1 - w) * X.uncred(pb, s, l, 8), s, l, k)
        u = P.d2(A.target_b.to_numpy(), p); o = P.d2_own(A.index, p)
        r[f"{l}_uff"] = round(u, 5); r[f"{l}_own"] = None if o is None else round(o, 5)
        tot_u += u * W[l]; tot_o += (o or 0) * W[l]
    r["tot_uff"] = round(tot_u / sum(W.values()), 5); r["tot_own"] = round(tot_o / sum(W.values()), 5)
    rows.append(r); print(r, flush=True)
o = pd.DataFrame(rows); o.to_csv(P.RESULTS / "holdout_judge.csv", index=False); print(o.to_string(index=False))
