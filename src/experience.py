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


# --- 20/10: credibilità senza i perili che non si ripetono ---
# Residui del freq×sev nel fabbricato: chi ha avuto solo sinistri di cedimento ha A/E 0,62/0,48 (2023/2024), chi ha avuto
# altri sinistri 1,14/1,18. Il cedimento dipende da terreno e siccità dell'anno, non dall'assicurato: non entra nel
# conteggio N della credibilità; gli attesi E si riducono della sua quota di sinistri. Verifica sulle OOF (k=8):
# blend fabbricato +0,00006 / +0,00006 nei due fold. Config: credibility.exclude_perils = {"buildings": ["subsidence"]}.
EXCL_DEFAULT = {"buildings": ["subsidence"]}
_cache_x = {}


def history_excl(line, excl):
    """N2 = sinistri passati della polizza senza i perili esclusi (AY < anno), H come in history(), quota dei conteggi tenuti."""
    key = (line, tuple(excl))
    if key in _cache_x:
        return _cache_x[key]
    h, _, sev = history(line)
    tr = P.load(line)
    perils = [c[3:] for c in tr.columns if c.startswith("pi_")]
    nn = tr[[f"n_{p}" for p in perils]].sum().to_numpy(); w = nn / nn.sum()
    keep = [p for p in perils if p not in excl]
    frac = float(w[[perils.index(p) for p in keep]].sum())
    tr["n_keep"] = tr[[f"n_{p}" for p in keep]].sum(axis=1)
    yr = tr.groupby(["policy_id", "year"])["n_keep"].sum().reset_index()
    te = P.load(line, "test")[["row_id", "policy_id", "year"]]
    allr = pd.concat([tr[["row_id", "policy_id", "year"]], te])
    yr2 = pd.concat([yr, te[["policy_id", "year"]].drop_duplicates().assign(n_keep=0.0)]).groupby(["policy_id", "year"], as_index=False)["n_keep"].sum()
    yr2 = yr2.sort_values(["policy_id", "year"])
    yr2["N2"] = yr2.groupby("policy_id")["n_keep"].transform(lambda s: s.shift().fillna(0).cumsum())
    n2 = allr.merge(yr2[["policy_id", "year", "N2"]], on=["policy_id", "year"], how="left").set_index("row_id")["N2"]
    _cache_x[key] = (h, n2, sev, frac)
    return _cache_x[key]


def apply_excl(pred, valid, line, k, excl=None):
    excl = EXCL_DEFAULT.get(line, []) if excl is None else excl
    if not excl:
        return apply(pred, valid, line, k)
    h, n2, sev, frac = history_excl(line, excl)
    pred = np.asarray(pred, float)
    H = h.reindex(valid["row_id"]).to_numpy(); N = n2.reindex(valid["row_id"]).to_numpy()
    E = H * pred / (valid["exposure"].to_numpy() * sev) * frac
    out = pred * (N + k) / (E + k)
    return out * pred.sum() / out.sum()


def uncred(pred_cred, valid, line, k):
    """Inverte apply(): dalla previsione con credibilità k alla previsione senza (esatta, iterazione sul riscalamento)."""
    h, n, sev = history(line)
    H = h.reindex(valid["row_id"]).to_numpy(); N = n.reindex(valid["row_id"]).to_numpy()
    a = H / (valid["exposure"].to_numpy() * sev); out = np.asarray(pred_cred, float); s = 1.0
    for _ in range(80):
        o = out / s; pr = k * o / (N + k - a * o); s_new = s * pr.sum() / out.sum()
        if abs(s_new - s) < 1e-13:
            break
        s = s_new
    return pr


# --- 22/10 notte: storia dell'ultimo anno completata con i sinistri tardivi attesi dai contatti ---
# Nel test la storia 2024 è ai denunciati entro il 31/12/2024 (−7% dei sinistri in auto). I contatti col servizio
# clienti della riga 2024 nascono anche dai sinistri non ancora denunciati: fra le righe senza denunce, P(tardivo)
# = 0,03% / 0,45% / 3,4% / 17% / 63% con 0/1/2/3/≥4 contatti (2022 e 2023 uguali; scripts/contacts_late.py).
# Qui: N += E[tardivi | contatti, denunciati] della riga 2024 della polizza (tabella 2022+2023), H riportata alla
# completezza piena (quella delle righe 2023, per mese). Simulazione nei fold (scripts/contacts_cred_bench.py):
# auto +0,34/+0,39 ufficiale, +0,23/+0,15 senza spalmatura (e·4) → tenuta; casa e contenuto negative → no.
CS_MAX, NR_MAX = 6, 2
_cache_fix = {}


def history_fix(line, last_year=2024, fit_years=(2022, 2023)):
    """N_fix e H_fix per le righe del test: storia completata sull'anno last_year."""
    key = (line, last_year)
    if key in _cache_fix:
        return _cache_fix[key]
    h, n, sev = history(line)
    tr = P.load(line); te = P.load(line, "test")
    tr["period_start"] = pd.to_datetime(tr.period_start); tr["period_end"] = pd.to_datetime(tr.period_end); tr["mese"] = tr.period_end.dt.month
    perils = [c[3:] for c in tr.columns if c.startswith("pi_")]
    nn = tr[[f"n_{p}" for p in perils]].sum().to_numpy(); w = nn / nn.sum()
    tr["pi_avg"] = tr[[f"pi_{p}" for p in perils]].to_numpy() @ w
    cl = pd.read_parquet(next(Path(P.CFG["raw"]).glob("**/claims.parquet")), columns=["policy_id", "line", "loss_date", "report_date"])
    cl = cl[cl.line == line]; cl["loss_date"] = pd.to_datetime(cl.loss_date); cl["report_date"] = pd.to_datetime(cl.report_date)
    c = cl.merge(tr[["row_id", "policy_id", "period_start", "period_end", "year"]], on="policy_id")
    c = c[(c.loss_date >= c.period_start) & (c.loss_date <= c.period_end)]
    c["cut"] = pd.to_datetime(c.year.astype(str) + "-12-31")
    tr["n_rep"] = c[c.report_date <= c.cut].groupby("row_id").size().reindex(tr.row_id).fillna(0).to_numpy()
    tr["n_late"] = c[c.report_date > c.cut].groupby("row_id").size().reindex(tr.row_id).fillna(0).to_numpy()
    tr["cs"] = tr.customer_service_contacts.clip(upper=CS_MAX); tr["nr"] = tr.n_rep.clip(upper=NR_MAX)
    tab = tr[tr.year.isin(fit_years)].groupby(["cs", "nr"])["n_late"].mean()
    pi_full = tr[tr.year == last_year - 1].groupby("mese")["pi_avg"].mean()
    last = tr[tr.year == last_year].copy()
    last["late_exp"] = tab.reindex(pd.MultiIndex.from_arrays([last.cs, last.nr])).fillna(0).to_numpy()
    last["dH"] = last.exposure * (pi_full.reindex(last.mese).to_numpy() - last.pi_avg)
    g = last.groupby("policy_id").agg(late_exp=("late_exp", "sum"), dH=("dH", "sum"))
    gg = g.reindex(te.policy_id).fillna(0)
    N_fix = pd.Series(n.reindex(te.row_id).to_numpy() + gg.late_exp.to_numpy(), index=te.row_id)
    H_fix = pd.Series(h.reindex(te.row_id).to_numpy() + gg.dH.to_numpy(), index=te.row_id)
    _cache_fix[key] = (N_fix, H_fix, sev, float(gg.late_exp.sum()), float(gg.dH.sum()))
    return _cache_fix[key]


def apply_fix(pred, valid, line, k):
    """come apply(), con la storia dell'ultimo anno completata (solo righe del test)."""
    N_fix, H_fix, sev, _, _ = history_fix(line)
    pred = np.asarray(pred, float)
    H = H_fix.reindex(valid["row_id"]).to_numpy(); N = N_fix.reindex(valid["row_id"]).to_numpy()
    out = pred * (N + k) / (H * pred / (valid["exposure"].to_numpy() * sev) + k)
    return out * pred.sum() / out.sum()
