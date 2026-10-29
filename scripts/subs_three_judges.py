"""Each uploaded submission rebuilt from the out-of-sample predictions (folds 2023 and 2024) and measured on the three judges of the
s3 configuration: official (target_b), own (no spreading), with actual large losses. D² by line and fold, Kaggle-style
weighted mean. Submissions 01–24 as in scripts/score_subs.py; 25–40 from the OOF pieces. Approximations
(missing OOF piece): corrected motor GBM without gross target (25–40); buildings without the subsidence-excluding credibility (29–40).
Output results_s3/subs_three_judges.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P  # noqa: E402
import score_subs as SS  # noqa: E402
W = SS.W
base = SS.rd("results_s3", "blend_opt").sort_values("row_id").reset_index(drop=True)
rid = base["row_id"]; line = base["line"].to_numpy(); fold = base["fold"].to_numpy(); y = base["target_b"].to_numpy()
idx = {k: np.where((line == k[0]) & (fold == k[1]))[0] for k in [(l, f) for l in W for f in ("F2023", "F2024")]}
def al(o, pred=None):
    p = o["pred"].to_numpy() if pred is None else pred
    return pd.Series(p, index=o["row_id"].to_numpy()).reindex(rid).to_numpy()
def judges(p):
    r = {}
    for nm, fn in (("ufficiale", lambda ii: P.d2(y[ii], p[ii])), ("senza_spalmatura", lambda ii: P.d2_own(rid.iloc[ii], p[ii])),
                   ("grandi_sinistri_veri", lambda ii: P.d2_unc(rid.iloc[ii], p[ii]))):
        r[nm] = sum(fn(ii) * W[l] for (l, f), ii in idx.items()) / (sum(W.values()) * 2)
    return r
S = {}
g = SS.rd("results", "glm_b"); b = SS.rd("results", "gbm_b")
S["sub_01"] = al(g, 0.5 * g["pred"].to_numpy() + 0.5 * b["pred"].to_numpy())
S["sub_03"] = al(SS.rd("results", "fs_peril_r4"))
s5 = SS.rd("results", "sub05").sort_values("row_id").reset_index(drop=True); S["sub_05"] = al(s5)
S["sub_06"] = al(s5, SS.by_line(s5, {"buildings": SS.pw(1.05)}))
fa = SS.rd("results", "fs2_all_ll_cred").set_index("row_id").reindex(s5["row_id"])["pred"].to_numpy()
q7 = SS.rd("results", "qwen_gbm73_cred").set_index("row_id").reindex(s5["row_id"])["pred"].to_numpy()
p7 = s5["pred"].to_numpy().copy(); mb = (s5["line"] == "buildings").to_numpy(); p7[mb] = 0.55 * fa[mb] + 0.45 * q7[mb]
s7 = s5.assign(pred=p7); S["sub_07"] = al(s7)
mod = pd.read_parquet(ROOT / "data/model/model_buildings.parquet", columns=["row_id", "period_end"]).set_index("row_id")
pe = pd.to_datetime(mod["period_end"].reindex(s7["row_id"]).to_numpy())
early = mb & ~((pe.month == 12) & (pe.day >= 30)); p8 = p7.copy(); p8[early] *= 0.85
for f in ("F2023", "F2024"):
    m = mb & (s7["fold"] == f).to_numpy(); p8[m] *= p7[m].sum() / p8[m].sum()
S["sub_08"] = al(s7, p8); S["sub_09"] = S["sub_07"]
for k, l, c in (("sub_10", "motor", 1.04), ("sub_11", "buildings", 1.04), ("sub_12", "contents", 1.04), ("sub_13", "motor", 0.976)):
    S[k] = al(s7, SS.by_line(s7, {l: SS.lv(c)}))
S["sub_15"] = al(SS.rd("results", "blend_opt")); S["sub_16"] = al(SS.rd("results_tro", "blend_opt")); S["sub_17"] = al(SS.rd("results_v3", "blend_opt"))
s3 = SS.rd("results_s3", "blend_opt"); S["sub_18"] = al(s3)
S["sub_19"] = al(s3, SS.by_line(s3, {"motor": SS.lv(581.9 / 568.9)})); S["sub_20"] = al(s3, SS.by_line(s3, {"buildings": SS.lv(1.02)}))
S["sub_21"] = al(s3, SS.by_line(s3, {"motor": SS.pw(0.95), "buildings": SS.pw(1.05), "contents": SS.pw(0.95)}))
a4 = SS.rd("results_s4", "fs2_v2_cred").set_index("row_id").reindex(s3["row_id"])["pred"].to_numpy()
qb = SS.rd("results_s3", "qwen_gbm73_cred").set_index("row_id").reindex(s3["row_id"])["pred"].to_numpy()
wts = json.loads((ROOT / "results_s3/blend_weights.json").read_text())["weights"]
S["sub_23"] = al(s3, SS.mix(s3, [a4, qb], {l: [wts[l]["fs2_v2_cred"], wts[l]["qwen_gbm73_cred"]] for l in wts}))
S["sub_24"] = al(s3, SS.by_line(s3, {"buildings": lambda p, m: (p ** 1.08) * p.sum() / (p ** 1.08).sum() * 1.02}))
# --- 25-40 from the pieces
R = ROOT / "results_s3"
A0 = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"].reindex(rid).to_numpy()
B0 = pd.read_parquet(R / "oof/qwen_gbm73_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"].reindex(rid).to_numpy()
def part(n, lines):
    out = np.full(len(rid), np.nan)
    for l in lines:
        for f in ("F2023", "F2024"):
            s = pd.read_parquet(R / f"oof_parts/{n}_{l}_{f}.parquet").set_index("row_id")["pred"]
            ii = idx[(l, f)]; out[ii] = s.reindex(rid.iloc[ii]).to_numpy()
    return out
Bfix = part("qwen_gbm73fix_cred", ["motor"]); Allr = part("fs2_v2_llr_cred", ["buildings"])
Bg = part("qwen_gbm73_gross_cred", ["buildings", "contents"])
Af = part("fs2_v2f_cred", ["motor", "buildings"]); Afs = part("fs2_v2fs_cred", ["motor"]); Afb = part("fs2_v2fb_cred", ["motor"])
mm, mbb, mc = line == "motor", line == "buildings", line == "contents"
def build(wm=0.9, wb=0.6, wc=0.8, am=A0, bm=B0, ab=A0, bb=B0, ac=A0, bc=B0):
    p = np.zeros(len(rid))
    p[mm] = wm * am[mm] + (1 - wm) * bm[mm]; p[mbb] = wb * ab[mbb] + (1 - wb) * bb[mbb]; p[mc] = wc * ac[mc] + (1 - wc) * bc[mc]
    return p
S["sub_25"] = build(wm=0.95); S["sub_26"] = A0.copy(); S["sub_27"] = B0.copy(); S["sub_28"] = build(wm=0.95, bb=Bg, bc=Bg)
k29 = dict(bm=Bfix, ab=Allr); S["sub_29"] = build(**k29)
S["sub_30"] = build(**dict(k29, am=Af, ab=Af)); k31 = dict(k29, am=Af); S["sub_31"] = build(**k31)
S["sub_32"] = build(**dict(k29, ab=Af)); S["sub_34"] = build(**dict(k31, wm=1.0)); S["sub_35"] = build(**dict(k31, wc=1.0))
S["sub_36"] = build(**dict(k31, bb=Bg)); S["sub_37"] = build(**dict(k31, bc=Bg)); S["sub_38"] = build(**dict(k31, am=Afs))
S["sub_39"] = build(**dict(k31, am=Afb)); S["sub_40"] = build(**dict(k31, am=Afb, bb=Bg))
rows = []
for k, p in S.items():
    assert np.isfinite(p).all(), k
    r = judges(p); r["sub"] = k; rows.append(r); print(k, {a: round(v, 5) for a, v in r.items() if a != "sub"}, flush=True)
pd.DataFrame(rows).to_csv(R / "subs_three_judges.csv", index=False)
