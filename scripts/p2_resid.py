"""Parte 2 (26/10): dove perde la FORMA di p2_sub_10. OOF di sub_10 (spec v4) sui fold K1–K5 del banco, poi per ogni variabile
A/E per livello con credibilità (K = costo atteso di 300 righe medie) stimato sugli altri 4 fold e applicato al quinto
(incrociato, livello del fold invariato). Guadagno di D² forma e own (e·4) e fold vinti su 5. Stesso test anche sul solo prior
del developed × livello (colonna 'prior'): quanto la risposta dell'emerging a ogni variabile differisce da quella del developed.
Uso: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_resid.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

DROP = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_raw",
        "customer_service_contacts", "renewal_decision", "written_premium", "email_domain", "construction_type_raw",
        "flag_exposure_fixed", "target_b", "cost_total", "rec_adj", "ult_capped", "ult25_capped", "year"}


def sub10(tr, va, l):
    if l == "motor":
        TR.CORRECTION_VARS, old = ["new_business", "gender"], TR.CORRECTION_VARS
        p = TR.src_glm(tr, va, l); TR.CORRECTION_VARS = old; return np.asarray(p, float)
    if l == "buildings":
        b = np.asarray(TR.src_level(tr, va, l), float); sc = np.asarray(TR.scratch(tr, va, l), float)
        return 0.9 * b + 0.1 * sc * b.sum() / sc.sum()
    p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    TR._PRIOR_OVERRIDE = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
    TR.CORRECTION_VARS, old = ["gender"], TR.CORRECTION_VARS
    p = TR.src_glm(tr, va, l); TR.CORRECTION_VARS = old; TR._PRIOR_OVERRIDE = None
    return np.asarray(p, float)


def binned(s):
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
        e = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, 7)[1:-1]))
        return pd.Series(np.where(s.isna(), "na", np.searchsorted(e, s.astype(float).fillna(0)).astype(str)), index=s.index)
    s = s.astype(object).where(s.notna(), "na").astype(str)
    vc = s.value_counts()
    return s.where(s.isin(vc[vc >= 200].index), "rare")


rows, oofs = [], []
for l in P.LINES:
    df = P.load(l)
    parts = []
    for fname, tr, va in folds(df):
        if not fname.startswith("K"):
            continue
        parts.append(pd.DataFrame({"row_id": va.row_id.to_numpy(), "fold": fname, "y": va[P.TARGET].to_numpy(),
                                   "sub10": sub10(tr, va, l), "prior": np.asarray(TR.src_level(tr, va, l), float)}))
        print(l, fname, flush=True)
    o = pd.concat(parts, ignore_index=True); o["line"] = l; oofs.append(o)
    X = df.set_index("row_id").reindex(o.row_id)
    cand = [c for c in X.columns if c not in DROP and not c.startswith(("cost_", "excess_", "ult_", "ult25_", "n_", "pi_"))]
    for base in ("sub10", "prior"):
        p0 = o[base].to_numpy(); y = o.y.to_numpy(); K = 300 * p0.mean()
        d0f, d0o = P.d2(y, p0 * y.sum() / p0.sum()), P.d2_own(o.row_id, p0)
        for v in cand:
            g = binned(X[v]).to_numpy()
            p1 = p0.copy(); wins = 0
            for f in o.fold.unique():
                m = (o.fold == f).to_numpy()
                t = pd.DataFrame({"g": g[~m], "y": y[~m], "p": p0[~m]}).groupby("g").sum()
                fac = (t.y + K) / (t.p + K) * (p0[~m].sum() / y[~m].sum())
                q = p0[m] * pd.Series(g[m]).map(fac).fillna(1.0).to_numpy(); q *= p0[m].sum() / q.sum()
                p1[m] = q
                wins += P.d2(y[m], q) > P.d2(y[m], p0[m])
            rows.append(dict(linea=l, base=base, var=v, livelli=len(set(g)),
                             forma=(P.d2(y, p1 * y.sum() / p1.sum()) - d0f) * 1e4, own=(P.d2_own(o.row_id, p1) - d0o) * 1e4,
                             vinti=f"{wins}/5"))
r = pd.DataFrame(rows)
out = Path(P.RESULTS); r.to_csv(out / "p2_resid.csv", index=False); pd.concat(oofs).to_parquet(out / "oof_sub10_K.parquet")
pd.set_option("display.width", 200)
for l in P.LINES:
    t = r[(r.linea == l)].pivot_table(index="var", columns="base", values=["forma", "own"]).round(1)
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.join(r[(r.linea == l) & (r.base == "sub10")].set_index("var")[["vinti", "livelli"]])
    print(f"\n=== {l}: guadagno incrociato e·4 di un A/E a una variabile (sub10 = sopra sub_10; prior = sopra il solo prior)")
    print(t.sort_values("forma_sub10", ascending=False).head(15).to_string())
