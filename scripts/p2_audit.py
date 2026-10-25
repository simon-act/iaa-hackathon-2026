"""Parte 2 — revisione indipendente di ipotesi e calcoli (25/10, richiesta di Simone: "rivedi tutte le ipotesi e tutti i calcoli e riprova").
Ricalcola dai file grezzi (data/raw_p2) con codice nuovo, senza le funzioni della pipeline, e confronta con quanto usato:
  A. dati: righe, row_id test = sample, esposizione dalle date, somme assicurate, sinistri esclusi;
  B. sinistri denunciati per linea × anno = tabella di sviluppo; target_b = Σ ultimate al 2025 + carico grandi sinistri;
  C. backtest delle curve di denuncia prese dal developed: AY 2023 visto al 31/12/2023 → previsione dei denunciati al
     31/12/2024, contro il vero (stesso test sul developed come metro del rumore);
  D. livello al 2025 per anno d'origine (2023 vs 2024) per linea e perilo.
Uso: python3 scripts/p2_audit.py  → stampa + results_p2/audit_*.csv"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, MOD, RES = ROOT / "data/raw_p2", ROOT / "data/interim_p2", ROOT / "data/model_p2", ROOT / "results_p2"
SRC_INT = ROOT / "data/interim_s3"
LINES = ["motor", "buildings", "contents"]
out = []


def check(area, cosa, valore, ok, nota=""):
    out.append(dict(area=area, controllo=cosa, valore=valore, esito="ok" if ok else "DA GUARDARE", nota=nota))
    print(f"[{'ok' if ok else '!!'}] {area} | {cosa}: {valore} {nota}")


# ---------- A. dati
samp = pd.read_parquet(RAW / "sample_submission.parquet")
te_ids = pd.concat([pd.read_parquet(RAW / f"test_{l}.parquet", columns=["row_id"]) for l in LINES]).row_id
check("A", "righe test = sample", f"{len(te_ids)} vs {len(samp)}", len(te_ids) == len(samp) and set(te_ids) == set(samp.row_id))
check("A", "row_id test unici", te_ids.is_unique, te_ids.is_unique)
for l in LINES:
    raw = pd.read_parquet(RAW / f"policies_{l}.parquet")
    cl = pd.read_parquet(INT / f"train_{l}.parquet")
    s, e = pd.to_datetime(raw.period_start), pd.to_datetime(raw.period_end)
    dy = np.where(s.dt.is_leap_year, 366, 365)
    expo = ((e - s).dt.days + 1) / dy
    m = cl.set_index("row_id").exposure.reindex(raw.row_id).to_numpy()
    diff = np.abs(expo.to_numpy() - m)
    check("A", f"{l}: esposizione pulita = giorni/anno dalle date", f"max scarto {diff.max():.4f}, righe >0.01: {(diff > 0.01).sum()}",
          (diff > 0.01).sum() == 0)
    bad = (np.abs(raw.exposure - expo) > 0.01).sum()
    check("A", f"{l}: esposizione grezza diversa dalle date", f"{bad} righe su {len(raw)}", True, "(corrette)")
    check("A", f"{l}: esposizione ≤ 1 e > 0", f"min {m.min():.4f} max {m.max():.4f}", (m > 0).all() and (m <= 1.0001).all())
    if "sum_insured" in raw:
        sc = cl.set_index("row_id").sum_insured.reindex(raw.row_id).to_numpy()
        med = np.nanmedian(raw.sum_insured)
        big = (raw.sum_insured > 50 * med).sum()
        check("A", f"{l}: somme assicurate ×1000 corrette", f"grezze >50×mediana {big}, pulite >50×mediana {(sc > 50 * med).sum()}, max pulita {np.nanmax(sc):,.0f}",
              (sc > 50 * med).sum() == 0)
        t = pd.read_parquet(RAW / f"test_{l}.parquet"); tc = pd.read_parquet(INT / f"test_{l}.parquet")
        check("A", f"{l} test: somme ×1000 corrette", f"grezze {(t.sum_insured > 50 * med).sum()}, pulite {(tc.sum_insured > 50 * med).sum()}",
              (tc.sum_insured > 50 * med).sum() == 0)
    tr = pd.read_parquet(RAW / f"test_{l}.parquet"); tc = pd.read_parquet(INT / f"test_{l}.parquet")
    s, e = pd.to_datetime(tr.period_start), pd.to_datetime(tr.period_end)
    ex = ((e - s).dt.days + 1) / 365
    d2 = np.abs(ex.to_numpy() - tc.set_index("row_id").exposure.reindex(tr.row_id).to_numpy())
    check("A", f"{l} test: esposizione dalle date", f"max scarto {d2.max():.4f}", (d2 > 0.01).sum() == 0)

clr = pd.read_parquet(RAW / "claims.parquet")
pol = pd.concat([pd.read_parquet(RAW / f"policies_{l}.parquet", columns=["row_id", "policy_id", "period_start", "period_end"]) for l in LINES])
pol["period_start"] = pd.to_datetime(pol.period_start); pol["period_end"] = pd.to_datetime(pol.period_end)
c = clr.copy(); c["loss_date"] = pd.to_datetime(c.loss_date); c["report_date"] = pd.to_datetime(c.report_date)
st = c.status.str.lower()
rej = st.isin(["repudiated", "declined", "rejected"])
c = c[~rej]
dup = c.duplicated(["policy_id", "loss_date", "peril", "incurred_to_date", "report_date"])
c = c[~dup]
pre = c.loss_date < "2023-01-01"; c = c[~pre]
mm = c.merge(pol, on="policy_id")
mm = mm[(mm.loss_date >= mm.period_start) & (mm.loss_date <= mm.period_end)]
nomatch = len(c) - mm.claim_id.nunique()
clean = pd.read_parquet(INT / "claims_clean.parquet")
check("A", "sinistri: grezzi → rifiutati / duplicati / prima del 2023 / senza polizza attiva",
      f"{len(clr)} → {rej.sum()} / {dup.sum()} / {pre.sum()} / {nomatch} → {mm.claim_id.nunique()} (pipeline {len(clean)})",
      mm.claim_id.nunique() == len(clean))
check("A", "sinistri: una sola riga di polizza per sinistro", f"{(mm.groupby('claim_id').size() > 1).sum()} multipli", (mm.groupby('claim_id').size() > 1).sum() == 0)
check("A", "sinistri: denuncia ≥ accadimento", f"{(c.report_date < c.loss_date).sum()} casi", (c.report_date < c.loss_date).sum() == 0)
check("A", "sinistri: valutazione", f"ultima denuncia {c.report_date.max().date()}, ultimo accadimento {c.loss_date.max().date()}", True)

# ---------- B. denunciato e target
dev = pd.read_parquet(INT / "development.parquet")
clean["net"] = clean.net_capped
g = clean.groupby(["line", "ay"]).net.sum()
gd = dev.groupby(["line", "ay"]).cost_rep.sum()
rel = (g / gd - 1).abs().max()
gd0 = dev[dev.rec_ratio == 0].groupby(["line", "peril", "ay"]).cost_rep.sum()
g0 = clean.groupby(["line", "peril", "ay"]).net.sum().reindex(gd0.index)
check("B", "denunciato netto tagliato = tabella di sviluppo (perili senza recuperi futuri)", f"max scarto relativo {(g0 / gd0 - 1).abs().max():.2e}", (g0 / gd0 - 1).abs().max() < 1e-6)
check("B", "denunciato con recuperi futuri sugli aperti (danni propri, furto)", f"scarto max {rel:.2%} = recuperi attesi sugli aperti", rel < 0.07)
net_raw = (clean.incurred_to_date - clean.recoveries)
check("B", "netto = incurred − recuperi", f"max scarto {np.abs(net_raw - clean.net_incurred).max():.4f}", np.abs(net_raw - clean.net_incurred).max() < 0.01)
lvl = pd.read_parquet(INT / "level_2025.parquet")
for l in LINES:
    md = pd.read_parquet(MOD / f"model_{l}.parquet")
    peri = [x[6:] for x in md.columns if x.startswith("ult25_") and x != "ult25_capped"]
    s_ult = md[[f"ult25_{p}" for p in peri]].sum(axis=1)
    ll = lvl.loc[l, "ll_load_2025"]
    recon = s_ult + ll * md.exposure
    d = np.abs(recon - md.target_b).max()
    check("B", f"{l}: target_b = Σ ult25 per perilo + carico grandi sinistri × esposizione", f"max scarto {d:.4f} (carico {ll:.2f}/anno)", d < 0.01)
    for y in (2023, 2024):
        s = md[md.year == y]
        n = clean[(clean.line == l) & (clean.ay == y)]
        rep_rows = s[[f"cost_{p}" for p in peri]].sum().sum()
        check("B", f"{l} {y}: Σ costo denunciato sulle righe = Σ sinistri", f"{rep_rows:,.0f} vs {n.net_capped.sum():,.0f}",
              abs(rep_rows / n.net_capped.sum() - 1) < 1e-6)
    tg = md.groupby("year").apply(lambda s: s.target_b.sum() / s.exposure.sum())
    print(f"   {l}: target_b per anno di esposizione 2023 {tg[2023]:.1f} | 2024 {tg[2024]:.1f}")

# ---------- C. backtest delle curve di denuncia (solo numeri: le riserve sui costi non hanno storia di valutazioni)
def ecdf_lag(cl, ays, horizon):
    """F(t) = quota denunciata entro t giorni, stimata sugli AY maturi con ritardo osservabile fino a 'horizon'."""
    x = cl[cl.ay.isin(ays)]
    lag = (x.report_date - x.loss_date).dt.days.to_numpy()
    lag = lag[lag <= horizon]
    t = np.arange(horizon + 1)
    return np.searchsorted(np.sort(lag), t, side="right") / len(lag)


def backtest(cl, ay, F, horizon):
    """visto al 31/12/AY: N̂ = osservati / π̄(1); previsti al 31/12/AY+1 = N̂·π̄(2); π̄ con accadimenti uniformi nell'anno."""
    days = pd.date_range(f"{ay}-01-01", f"{ay}-12-31")
    v1, v2 = pd.Timestamp(f"{ay}-12-31"), pd.Timestamp(f"{ay + 1}-12-31")
    p1 = F[np.minimum((v1 - days).days, horizon)].mean(); p2 = F[np.minimum((v2 - days).days, horizon)].mean()
    x = cl[cl.ay == ay]
    o1 = (x.report_date <= v1).sum(); o2 = (x.report_date <= v2).sum()
    return o1, o1 / p1 * p2, o2


src = pd.read_parquet(SRC_INT / "claims_clean.parquet")
rows = []
H = 1095
for (l, p), x in clean.groupby(["line", "peril"]):
    xs = src[(src.line == l) & (src.peril == p)]
    F = ecdf_lag(xs, [2021], H)           # AY 2021 del developed: ritardi osservabili fino a ~3 anni
    o1, e2, o2 = backtest(x, 2023, F, H)
    so1, se2, so2 = backtest(xs, 2023, F, H)
    rows.append(dict(linea=l, perilo=p, visti_2023=o1, previsti_fine_2024=round(e2, 1), veri_fine_2024=o2, AE=round(o2 / e2, 3),
                     dev_AE_stesso_test=round(so2 / se2, 3), sd_poisson_AE=round(np.sqrt(max(e2 - o1, 1)) / e2, 3)))
bt = pd.DataFrame(rows)
print(bt.to_string(index=False))
for _, r in bt.iterrows():
    z = (r.veri_fine_2024 - r.previsti_fine_2024) / np.sqrt(max(r.previsti_fine_2024 - r.visti_2023, 1))
    check("C", f"{r.linea}/{r.perilo}: denunciati AY2023 a fine 2024, veri/previsti", f"{r.AE} (developed stesso test {r.dev_AE_stesso_test}, z={z:+.1f})",
          abs(z) < 2.5)
bt.to_csv(RES / "audit_backtest_denuncia.csv", index=False)

# ---------- D. livello per anno d'origine
on = pd.read_parquet(INT / "development_onlevel.parquet")
piv = on.pivot_table(index=["line", "peril"], columns="ay", values="bc_onlevel")
piv["rapporto_24_23"] = piv[2024] / piv[2023]
print(piv.round(2).to_string())
for l in LINES:
    t = on[on.line == l].groupby("ay").apply(lambda s: (s.bc_onlevel).sum())
    check("D", f"{l}: costo per anno al livello 2025, 2024/2023", f"{t[2023]:.1f} → {t[2024]:.1f} ({t[2024] / t[2023]:.2f})", 0.8 < t[2024] / t[2023] < 1.25)
pd.DataFrame(out).to_csv(RES / "audit_controlli.csv", index=False)
print(f"\n{sum(o['esito'] == 'ok' for o in out)}/{len(out)} ok")
