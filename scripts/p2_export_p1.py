"""Parte 2 (27/10, richiesta di Simone: "testare i migliori modelli della Parte 1 sulla Parte 2"): esporta come prior del developed
una variante di modello della Parte 1 (senza la credibilità finale sulla storia, che nell'emerging peggiora: bench4), con la stessa
selezione di variabili comuni di transfer.export_source. Varianti:
  v2      = fs2_v2 (casa con premio relativo; auto = fs2_all_ll perché il chilometraggio manca nell'emerging)   [sub_26]
  v2f     = v2 + frequenza per perilo da griglia (FREQ_CHOSEN)                                                    [sub_30]
  v2fs    = v2f + severità auto per perilo (SEV_CHOSEN: lesioni costante)                                        [sub_38/39]
  v2fsllr = v2fs + grandi sinistri ripartiti sul rischio                                                          [sub_45]
  gbm73   = GBM di Qwen (qwen_gbm73), per la miscela                                                              [sub_25]
  v2prem  = freq×sev con il premio relativo in frequenza e severità anche in auto (28/10)
Uscita: <results nuovo>/source_pred_<linea>_p1<variante>.parquet (mu_src, parti per perilo se il modello le produce).
Uso: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export_p1.py <linea> <variante> configs/parte2_llc.json [frazione=1]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M, market_config as MC  # noqa: E402

line, var, new_cfg = sys.argv[1], sys.argv[2], MC.load(sys.argv[3]); FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
t0 = time.time()
src = TR.harmonize(P.load(line))
import os
_RANGE = os.environ.get("P1_RANGE")   # 28/10: "lo-hi" = polizze con hash % 100 in [lo, hi) (metà disgiunte per il bagging)
if _RANGE:
    lo, hi = map(int, _RANGE.split("-")); hh = pd.util.hash_array(src["policy_id"].to_numpy() + "_c") % 100
    src = src[(hh >= lo) & (hh < hi)].reset_index(drop=True); FRAC = (hi - lo) / 100
elif FRAC < 1:
    src = src[pd.util.hash_array(src["policy_id"].to_numpy() + "_c") % 100 < FRAC * 100].reset_index(drop=True)
new_tr = pd.read_parquet(Path(new_cfg["model_dir"]) / f"model_{line}.parquet")
new_te = pd.read_parquet(Path(new_cfg["model_dir"]) / f"test_{line}.parquet")
common = [c for c in M._features(src, line) if c in new_tr.columns and c in new_te.columns]


def overlap(c):
    if pd.api.types.is_numeric_dtype(src[c]) and pd.api.types.is_numeric_dtype(new_te[c]):
        return 1.0
    seen = set(src[c].dropna().astype(str).unique()); v = new_te[c].dropna().astype(str)
    return float(v.isin(seen).mean()) if len(v) else 1.0


common = [c for c in common if overlap(c) >= 0.5]
M.FEATURE_WHITELIST = set(common) | {"prem_rel"}
import os
if os.environ.get("P1_THREADS"):   # 28/10: più thread per stare nel limite di tempo (risultati quasi identici, non bit a bit)
    for d_ in (M.FREQ_PARAMS, M.SEV_PARAMS):
        d_["num_threads"] = int(os.environ["P1_THREADS"])
both = pd.concat([new_tr, new_te], ignore_index=True)
fn = {"v2": M.fs2_v2, "gbm73": M.qwen_gbm73}
def v2f(tr, va, l):
    M._FREQ_OVERRIDE = M.FREQ_CHOSEN
    try:
        return M.fs2_v2(tr, va, l)
    finally:
        M._FREQ_OVERRIDE = {}
def v2fs(tr, va, l):
    M._SEV_PERIL = M.SEV_CHOSEN
    try:
        return v2f(tr, va, l)
    finally:
        M._SEV_PERIL = {}
def v2fsllr(tr, va, l):
    M._LL_RISK = True
    try:
        return v2fs(tr, va, l)
    finally:
        M._LL_RISK = False
def v2prem(tr, va, l):   # 28/10: freq×sev con il premio relativo anche in auto (in fs2_v2 era solo in casa)
    M.SEV_DRIVERS[l] = M.SEV_DRIVERS[l] + ["prem_rel"]
    try:
        return M.fs2_all_ll(M._prem(tr), M._prem(va), l)
    finally:
        M.SEV_DRIVERS[l] = [c for c in M.SEV_DRIVERS[l] if c != "prem_rel"]
fn.update(v2f=v2f, v2fs=v2fs, v2fsllr=v2fsllr, v2prem=v2prem)
M._fs2.last_parts = None
pred = np.asarray(fn[var](src, both, line), float)
out = pd.DataFrame({"row_id": both["row_id"].to_numpy(), "mu_src": pred})
lp = getattr(M._fs2, "last_parts", None)
gb = pd.read_parquet(Path(new_cfg["results"]) / f"source_pred_{line}.parquet").set_index("row_id").reindex(out.row_id)
if lp and var != "gbm73":
    for per, (f_va, s_va) in lp["perils"].items():
        out[f"f_{per}"] = f_va; out[f"s_{per}"] = s_va
    out["ll_src"] = lp["ll"]; out["c_src"] = lp["c"]
else:   # parti per perilo prese dal prior attuale (servono solo ai livelli per perilo e ai numeri attesi)
    for c in gb.columns:
        if c != "mu_src":
            out[c] = gb[c].to_numpy()
out.to_parquet(Path(new_cfg["results"]) / f"source_pred_{line}_p1{var}{('_r' + _RANGE.replace('-', '_')) if _RANGE else ('' if FRAC == 1 else '_s' + str(int(FRAC * 100)))}.parquet", index=False)
print(f"{line} {var}: corr log col prior attuale {np.corrcoef(np.log(pred), np.log(gb.mu_src.to_numpy()))[0, 1]:.4f}, "
      f"rapporto totali {pred.sum() / gb.mu_src.sum():.3f} ({time.time() - t0:.0f}s, {len(src)} righe)", flush=True)
