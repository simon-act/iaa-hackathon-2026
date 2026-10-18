"""Credibilità sull'esperienza della polizza (experience rating, Bühlmann–Straub su conteggi).
Ipotesi: oltre alle variabili, ogni polizza ha un rischio latente (frailty) che si vede dai suoi sinistri passati.
Modificatore per riga: M = (N_hist + k) / (E_hist + k)
  N_hist = sinistri passati della polizza nella linea (AY < anno della riga, riportati alla data dei dati)
  E_hist = sinistri attesi su quella storia = Σ_storia(esposizione × completezza) × frequenza attuale della riga
           (frequenza attuale = previsione / (esposizione × severità media di linea))
  k = sinistri attesi a cui la storia vale il 50% (k grande = poca credibilità), scelto per linea con validazione incrociata.
Poi la previsione viene ribilanciata per non spostare il livello totale.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402

_cache = {}


def history(line):
    """per ogni riga (train e test): N_hist e H = Σ esposizione×completezza delle righe precedenti della stessa polizza."""
    if line in _cache:
        return _cache[line]
    tr = P.load(line); te = P.load(line, "test")
    perils = [c[3:] for c in tr.columns if c.startswith("pi_")]
    nn = tr[[f"n_{p}" for p in perils]].sum().to_numpy(); w = nn / nn.sum()
    tr["pi_avg"] = tr[[f"pi_{p}" for p in perils]].to_numpy() @ w
    sev = tr["target_b"].sum() / (tr[[f"n_{p}" for p in perils]].to_numpy() / tr[[f"pi_{p}" for p in perils]].clip(lower=0.05).to_numpy()).sum()
    tr["eh"] = tr["exposure"] * tr["pi_avg"]
    allr = pd.concat([tr[["row_id", "policy_id", "year", "eh"]], te[["row_id", "policy_id", "year"]].assign(eh=np.nan)])
    allr = allr.sort_values(["policy_id", "year"])
    allr["H"] = allr.groupby("policy_id")["eh"].transform(lambda s: s.shift().fillna(0).cumsum())
    h = allr.set_index("row_id")["H"]
    np_ = pd.concat([tr.set_index("row_id")["pol_prev_n"], te.set_index("row_id")["pol_prev_n"]])
    _cache[line] = (h, np_, sev)
    return _cache[line]


def modifier(row_id, exposure, pred, line, k):
    h, n, sev = history(line)
    H = h.reindex(row_id).to_numpy(); N = n.reindex(row_id).to_numpy()
    lam = pred / (np.asarray(exposure) * sev)
    E = H * lam
    M = (N + k) / (E + k)
    return M


def apply(pred, valid, line, k):
    pred = np.asarray(pred, float)
    M = modifier(valid["row_id"], valid["exposure"].to_numpy(), pred, line, k)
    out = pred * M
    return out * pred.sum() / out.sum()


if __name__ == "__main__":
    # tuning di k sulle previsioni out-of-fold di un modello: python -I experience.py <modello>
    name = sys.argv[1]
    o = pd.read_parquet(P.RESULTS / "oof" / f"{name}.parquet")
    ks = [0.25, 0.5, 1, 2, 4, 8, 16, 1e9]
    res = {}
    for line in P.LINES:
        for f in ("F2023", "F2024"):
            s = o[(o.line == line) & (o.fold == f)]
            res[(line, f)] = {k: P.d2(s.target_b, apply(s.pred.to_numpy(), s, line, k)) for k in ks}
        print(line, {f: {k: round(v, 5) for k, v in res[(line, f)].items()} for f in ("F2023", "F2024")}, flush=True)
