"""Part 2 (26/10): own level ONLY for some perils, the rest at line level, on top of p2_sub_10.
Rationale (results_p2/p2_bench5_livelli.csv): buildings storm 0.94–1.02 above the line level; motor glass_hail and TPP
below. Peril-level for ALL perils had been rejected (buildings −9.5, motor −3.5, 2/7).
Construction: sub_10 prior × (1 + Σ_{own p} share_p × (r_p − 1)), share_p = f_p·s_p / mu_src of the row,
r_p = (a_p·b_p)/(a_line·b_line) from TR.src_peril (credibility k toward the line, estimated on the fold's train only).
Then identical to sub_10 (spec v4): motor GLM gender+nb with offset; buildings level + 0.1 scratch.
Decision fixed in advance: k = 100/200 for storm, storm+subsidence, glass+TPP; other k values and single perils = sensitivity.
Rule: adopted if it wins ≥ 5/7 checks on shape judge AND own.  Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench10.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

VARS = {"motor": ["new_business", "gender"], "buildings": None}
W0 = {"motor": 0.0, "buildings": 0.1}
KS = [(25, 50), (100, 200), (400, 800)]
SETS = {"buildings": {"storm": ["storm"], "storm+subs": ["storm", "subsidence"], "nessuno": []},
        "motor": {"glass+tpp": ["glass_hail", "third_party_prop"], "glass": ["glass_hail"], "tpp": ["third_party_prop"], "nessuno": []}}
DECISIONE = {"storm k=100/200", "storm+subs k=100/200", "glass+tpp k=100/200"}


def corrected(tr, va, l, override=None):
    TR._PRIOR_OVERRIDE = override
    try:
        if VARS[l]:
            old, TR.CORRECTION_VARS = TR.CORRECTION_VARS, VARS[l]
            try:
                return np.asarray(TR.src_glm(tr, va, l), float)
            finally:
                TR.CORRECTION_VARS = old
        return np.asarray(TR.src_level(tr, va, l), float)
    finally:
        TR._PRIOR_OVERRIDE = None


rows, logs = [], []
for l in ("buildings", "motor"):
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        sc = np.asarray(TR.scratch(tr, va, l), float) if W0[l] else None
        def fin(b):
            return (1 - W0[l]) * b + W0[l] * sc * b.sum() / sc.sum() if W0[l] else b
        preds = {"sub_10": fin(corrected(tr, va, l))}
        for kf, ks in KS:
            ov0, r, lg = TR.peril_own_override(tr, va, l, [], kf, ks)
            logs.append(dict(linea=l, fold=fname, k=f"{kf}/{ks}", **{p: round(v, 4) for p, v in r.items()},
                             **{f"n_{p}": v["n"] for p, v in lg["perils"].items()}))
            for nome, own in SETS[l].items():
                ov = TR.peril_own_override(tr, va, l, own, kf, ks)[0]
                preds[f"{nome} k={kf}/{ks}"] = fin(corrected(tr, va, l, ov))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "done", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_10"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
ctl = r[r.variante.str.startswith("nessuno")][["d_forma", "d_own", "d_unc"]].abs().to_numpy().max()
print(f"control (zero own perils = sub_10): max |Δ| = {ctl:.2e}")
assert ctl < 1e-6, "the control does not reproduce sub_10"
out = Path(P.RESULTS)
r.to_csv(out / "p2_bench10.csv", index=False); pd.DataFrame(logs).to_csv(out / "p2_bench10_livelli.csv", index=False)
pd.set_option("display.width", 220)
s = r[~r.variante.isin(["sub_10"]) & ~r.variante.str.startswith("nessuno")].groupby(["linea", "variante"]).agg(
    forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"),
    pos_unc=("d_unc", lambda x: f"{(x > 0).sum()}/{len(x)}"))
s["decisione"] = [v in DECISIONE for _, v in s.index]
print("Δ e·4 vs sub_10"); print(s.round(2).to_string())
for c in ("d_forma", "d_own"):
    print(c); print(r[r.variante.isin(DECISIONE)].pivot_table(index=["linea", "variante"], columns="fold", values=c).round(2).to_string())
print("r_p (peril level / line level), k=100/200"); print(pd.DataFrame(logs).query("k=='100/200'").round(3).to_string())
