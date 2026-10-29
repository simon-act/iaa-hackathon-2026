"""21/10 evening: credibility of the history excluding 'weather' perils (event claims, not the policy's own risk).
Hint from hist_scan.py → in buildings, policies whose customer also has claims on other lines (same event) have A/E 0.73–0.80,
those with their own history A/E 1.1–1.3. Exclusion and k variants by line, applied to both components of the
sub_41 blend (uncred k=8 → apply_excl), line Δ e·4 on the own (no spreading) and official judges, two folds. Output results_s3/cred_excl_grid.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS
part = lambda n, l, f: pd.read_parquet(R / f"oof_parts/{n}_{l}_{f}.parquet").set_index("row_id")["pred"]
ref = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "line", "fold", "year", "exposure", "target_b"])
QB = pd.read_parquet(R / "oof/qwen_gbm73_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]
SPEC = {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_cred", 0.9, []), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, ["subsidence"]),
        "contents": ("fs2_v2_sevmix_cred", None, 0.8, [])}
VAR = {"motor": [[], ["glass_hail"], ["glass_hail", "theft"]],
       "buildings": [["subsidence"], ["subsidence", "flood"], ["subsidence", "storm"], ["subsidence", "flood", "storm"], []],
       "contents": [[], ["flood"], ["flood", "theft"]]}
rows = []
for l, (a, b, w, ex0) in SPEC.items():
    for f in P.FOLDS:
        s = ref[(ref.line == l) & (ref.fold == f)].copy(); rid = s.row_id
        pa = X.uncred(part(a, l, f).reindex(rid).to_numpy(), s, l, 8); pb = X.uncred((part(b, l, f) if b else QB).reindex(rid).to_numpy(), s, l, 8)
        def blend(excl, k):
            return w * X.apply_excl(pa, s, l, k, excl) + (1 - w) * X.apply_excl(pb, s, l, k, excl)
        p0 = blend(ex0, 8); o0, u0 = P.d2_own(rid, p0), P.d2(s.target_b, p0)
        for excl in VAR[l]:
            for k in (4, 8, 12):
                p1 = blend(excl, k)
                rows.append(dict(line=l, fold=f, excl="+".join(excl) or "-", k=k, own=(P.d2_own(rid, p1) - o0) * 1e4, uff=(P.d2(s.target_b, p1) - u0) * 1e4))
        print(l, f, "done", flush=True)
r = pd.DataFrame(rows); r.to_csv(R / "cred_excl_grid.csv", index=False)
print(r.pivot_table(index=["line", "excl", "k"], columns="fold", values=["own", "uff"]).round(2).to_string())
