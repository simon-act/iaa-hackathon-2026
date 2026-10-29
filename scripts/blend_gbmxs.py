"""22/10 evening: GBM with the relative deductible (xs_rel) inside the sub_41 blend, Δ e·4 on the three judges, two folds.
Motor: qwen_gbm73fix_cred → qwen_gbm73fix_xs_cred; buildings and contents: sub_41 GBM → qwen_gbm73_gross_xs_cred
(for contents the comparison with qwen_gbm73_gross_cred, same target construction, is also shown).
Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/blend_gbmxs.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS
QB = pd.read_parquet(R / "oof/qwen_gbm73_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]


def part(n, l, f):
    return QB if n == "qwen_gbm73_cred" else pd.read_parquet(R / f"oof_parts/{n}_{l}_{f}.parquet").set_index("row_id")["pred"]


ref = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "line", "fold", "exposure", "target_b"])
# (freq×sev, GBM, variants (freq×sev, GBM), weight, subsidence exclusion)
SPEC = {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_cred", [("fs2_v2fb_cred", "qwen_gbm73fix_xs_cred"), ("fs2_v2fb_xs_cred", "qwen_gbm73fix_cred"),
                                                          ("fs2_v2fb_xs_cred", "qwen_gbm73fix_xs_cred"),
                                                          ("fs2_v2fb_2ph_cred", "qwen_gbm73fix_cred")], 0.9, False),
        "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", [("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred"), ("fs2_v2_llr_xs_cred", "qwen_gbm73_gross_cred"),
                                                                   ("fs2_v2_llr_xs_cred", "qwen_gbm73_gross_xs_cred"),
                                                                   ("fs2_v2_llr_cred", "qwen_gbm73_gross_siinv_cred"), ("fs2_v2_llr_cred", "qwen_gbm73_gross_rnd_cred"), ("fs2_v2_llr_cred", "qwen_gbm73_gross_rnd12_cred"), ("fs2_v2_llr_cred", "qwen_gbm73_gross_rnd13_cred"), ("fs2_v2_llr_cred", "qwen_gbm73_gross_rnd14_cred")], 0.6, True),
        "contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", [("fs2_v2_sevmix_cred", "qwen_gbm73_gross_cred"), ("fs2_v2_sevmix_cred", "qwen_gbm73_gross_xs_cred"),
                                                               ("fs2_v2_sevmix_xs_cred", "qwen_gbm73_cred"), ("fs2_v2_sevmix_xs_cred", "qwen_gbm73_gross_xs_cred")], 0.8, False)}
rows = []
for l, (a, b, alts, w, ex) in SPEC.items():
    for f in P.FOLDS:
        s = ref[(ref.line == l) & (ref.fold == f)]; rid = s.row_id
        def ok(n):
            return n == "qwen_gbm73_cred" or (R / f"oof_parts/{n}_{l}_{f}.parquet").exists()

        def blend(an, bn):
            pa, pb = part(an, l, f).reindex(rid).to_numpy(), part(bn, l, f).reindex(rid).to_numpy()
            if ex:
                pa = X.apply_excl(X.uncred(pa, s, l, 8), s, l, 8); pb = X.apply_excl(X.uncred(pb, s, l, 8), s, l, 8)
            return w * pa + (1 - w) * pb
        p0 = blend(a, b); j0 = (P.d2(s.target_b, p0), P.d2_own(rid, p0), P.d2_unc(rid, p0))
        for an, bn in alts:
            if not (ok(an) and ok(bn)):
                print(l, f, "missing", an, bn); continue
            p = blend(an, bn); j = (P.d2(s.target_b, p), P.d2_own(rid, p), P.d2_unc(rid, p))
            rows.append(dict(line=l, fold=f, fs=an, gbm=bn, uff=(j[0] - j0[0]) * 1e4, own=(j[1] - j0[1]) * 1e4, unc=(j[2] - j0[2]) * 1e4))
r = pd.DataFrame(rows); r.to_csv(R / "blend_gbmxs.csv", index=False)
print(r.round(2).to_string(index=False))
