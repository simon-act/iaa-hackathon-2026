"""Parte 2 (28/10): ablazione di p2_sub_20. Ogni pezzo è stato scelto con il prior di allora; con il prior nuovo (casa: fs2_v2 di
sub_26) qualcuno potrebbe non servire più. Si toglie un pezzo alla volta; se TOGLIERLO vince ≥ 5/7 su forma E own, il pezzo esce.
Uso: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench23.py <linea>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
l = sys.argv[1]
TR.PRIOR_SUFFIX = "" if l == "motor" else "_p1v2"


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def variants(tr, va):
    out = {}
    if l == "motor":
        a1 = lambda d: glm(tr, d, ["new_business", "gender"]); a2 = lambda d: glm(tr, d, ["new_business", "gender", "fuel_type", "cover_type"])
        h = lambda d: 0.5 * a1(d) + 0.5 * a2(d)
        g = TR.gbm_freq_correction(tr, va, l, h(tr)); fp = TR.freq_correction(tr, va, l, ["prem_rel"])
        out["sub_20"] = h(va) * g * fp
        out["senza fuel/cover"] = a1(va) * TR.gbm_freq_correction(tr, va, l, a1(tr)) * fp
        out["senza GBM numero sinistri"] = h(va) * fp
        out["senza premio"] = h(va) * g
    elif l == "buildings":
        sc = np.asarray(TR.scratch(tr, va, l), float)
        mix = lambda p, w: (1 - w) * p + w * sc * p.sum() / sc.sum()
        ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
        st = glm(tr, va, None, ov); lv = glm(tr, va, None)
        out["sub_20"] = mix(st, 0.1); out["senza storm"] = mix(lv, 0.1); out["senza 10% solo emerging"] = st
    else:
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        wfh = TR.freq_correction(tr, va, l, ["works_from_home"])
        out["sub_20"] = glm(tr, va, ["gender"], ov) * wfh
        out["senza genere"] = glm(tr, va, None, ov) * wfh
        out["senza lavoro da casa"] = glm(tr, va, ["gender"], ov)
        out["senza livelli per perilo"] = glm(tr, va, ["gender"]) * wfh
    return out


rows = []
df = P.load(l); df = M._prem(df) if l == "motor" else df
for fname, tr, va in folds(df):
    y = va[P.TARGET].to_numpy()
    for k, p in variants(tr, va).items():
        rows.append(dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_20"].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / f"p2_bench23_{l}.csv", index=False)
print(l, "(Δ = togliere il pezzo; positivo = il pezzo fa male)")
print(r[r.variante != "sub_20"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
      pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/7"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/7")).round(2).to_string())
