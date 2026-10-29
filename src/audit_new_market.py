"""Part 2 — automated audit of a new market, compared with Part 1.

Answers the judges' question "how did you work out what changed?" in a reproducible way:
    python -I audit_new_market.py <raw_nuovo> <raw_parte1> <out.md> [--lines motor,buildings,contents]

For each table: rows, columns missing / new / with changed type relative to Part 1.
For each common column: missing rate, PSI between the two markets (numeric in 10 bins, categorical by level),
new levels of categoricals, quantiles. Data-quality checks already seen in Part 1 (exposure ≠ dates, sum
insured in wrong units, IDs without leading zeros, undocumented codes, duplicated / repudiated claims / claims before
inception, reporting lag). Technical levels by line and peril: frequency, severity, burning cost, open share,
large losses, with the new / Part 1 ratio. Writes a markdown file with a "what changed" section sorted by PSI.
No decision is taken here: the file is the input for the discussion and for configs/<market>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LINES = ["motor", "buildings", "contents"]
ID_COLS = {"row_id", "policy_id", "customer_id", "address_id", "claim_id", "repairer_id", "third_party_id", "surname"}


def psi(a: pd.Series, b: pd.Series, bins=10) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and a.nunique() > bins:
        edges = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a.astype(float), edges)[0] + 1.0
        pb = np.histogram(b.astype(float), edges)[0] + 1.0
    else:
        va, vb = a.astype(str).value_counts(), b.astype(str).value_counts()
        k = va.index.union(vb.index)
        pa = va.reindex(k).fillna(0).to_numpy() + 1.0
        pb = vb.reindex(k).fillna(0).to_numpy() + 1.0
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def load_tables(raw: Path, lines):
    t = {}
    for name in ["policies", "test"]:
        for line in lines:
            f = raw / f"{name}_{line}.parquet"
            if f.exists():
                t[f"{name}_{line}"] = pd.read_parquet(f)
    for name in ["claims", "customers", "addresses", "weather", "sample_submission"]:
        f = raw / f"{name}.parquet"
        if f.exists():
            t[name] = pd.read_parquet(f)
    return t


def compare_schema(new: dict, old: dict, md: list):
    md.append("## 1. Tabelle e colonne\n")
    md.append("| Tabella | Righe nuovo | Righe Parte 1 | Colonne mancanti nel nuovo | Colonne nuove | Tipo cambiato |")
    md.append("|---|---|---|---|---|---|")
    for k in sorted(set(new) | set(old)):
        if k not in new:
            md.append(f"| {k} | **assente** | {len(old[k])} | tutte | | |"); continue
        if k not in old:
            md.append(f"| {k} | {len(new[k])} | **assente** | | tutte ({len(new[k].columns)}) | |"); continue
        a, b = new[k], old[k]
        miss = sorted(set(b.columns) - set(a.columns)); extra = sorted(set(a.columns) - set(b.columns))
        typ = [c for c in set(a.columns) & set(b.columns)
               if pd.api.types.is_numeric_dtype(a[c]) != pd.api.types.is_numeric_dtype(b[c])]
        md.append(f"| {k} | {len(a)} | {len(b)} | {', '.join(miss) or '–'} | {', '.join(extra) or '–'} | {', '.join(sorted(typ)) or '–'} |")
    md.append("")


def compare_columns(new: dict, old: dict, md: list, top=40):
    rows = []
    for k in sorted(set(new) & set(old)):
        a, b = new[k], old[k]
        for c in a.columns:
            if c in ID_COLS or c not in b.columns:
                continue
            x, y = a[c], b[c]
            if pd.api.types.is_datetime64_any_dtype(x) or str(x.dtype).startswith("date") or "date" in c or c.startswith("period_"):
                continue   # dates are read from the data-quality checks (years covered), not via PSI
            r = {"tabella": k, "colonna": c, "mancanti nuovo %": round(100 * x.isna().mean(), 2),
                 "mancanti P1 %": round(100 * y.isna().mean(), 2), "PSI": round(psi(y, x), 4)}
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
                qa, qb = x.quantile([.01, .5, .99]).to_numpy(), y.quantile([.01, .5, .99]).to_numpy()
                r["mediana nuovo / P1"] = round(float(qa[1] / qb[1]), 3) if qb[1] else float("nan")
                r["p1 / p99 nuovo"] = f"{qa[0]:.3g} / {qa[2]:.3g}"
                r["p1 / p99 P1"] = f"{qb[0]:.3g} / {qb[2]:.3g}"
                r["livelli nuovi"] = ""
            else:
                la, lb = set(x.dropna().astype(str)), set(y.dropna().astype(str))
                r["mediana nuovo / P1"] = float("nan"); r["p1 / p99 nuovo"] = f"{len(la)} livelli"; r["p1 / p99 P1"] = f"{len(lb)} livelli"
                nl = sorted(la - lb); r["livelli nuovi"] = ", ".join(nl[:6]) + (" …" if len(nl) > 6 else "")
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("PSI", ascending=False)
    md.append("## 2. Cosa è cambiato nelle variabili (ordinato per PSI; > 0,25 = distribuzione molto diversa, 0,1–0,25 = diversa)\n")
    md.append(df.head(top).to_markdown(index=False))
    md.append("\n<details><summary>tutte le colonne</summary>\n\n" + df.to_markdown(index=False) + "\n</details>\n")
    return df


def quality_checks(t: dict, lines, md: list, label: str):
    md.append(f"## 3. Controlli di qualità ({label})\n")
    out = {}
    for line in lines:
        k = f"policies_{line}"
        if k not in t:
            continue
        p = t[k].copy()
        ps, pe = pd.to_datetime(p["period_start"]), pd.to_datetime(p["period_end"])
        days = (pe - ps).dt.days + 1
        r = {"righe": len(p), "anni": f"{ps.dt.year.min()}–{ps.dt.year.max()}",
             "polizze": p["policy_id"].nunique(),
             "esposizione ≠ giorni/365 (quota)": round(float(((p["exposure"] - days / 365).abs() > 0.5 / 365).mean()), 4),
             "esposizione > 1,5 (giorni?)": int((p["exposure"] > 1.5).sum()),
             "customer_id lunghezza min/max": f"{p['customer_id'].astype(str).str.len().min()}/{p['customer_id'].astype(str).str.len().max()}",
             "periodo che finisce prima di dicembre (quota)": round(float((~((pe.dt.month == 12) & (pe.dt.day >= 30))).mean()), 3),
             "row_id duplicati": int(p["row_id"].duplicated().sum())}
        if "sum_insured" in p.columns:
            si = p["sum_insured"]; med = si.median()
            r["sum_insured mediana"] = round(float(med), 0)
            r["sum_insured > 100× mediana (unità?)"] = int((si > 100 * med).sum())
            r["sum_insured ≤ 0 o mancante"] = int((si.isna() | (si <= 0)).sum())
        for c in ["construction_type", "cover_type", "tenure", "channel", "payment_method"]:
            if c in p.columns:
                r[f"livelli {c}"] = ", ".join(sorted(p[c].dropna().astype(str).unique())[:12])
        out[line] = r
    md.append(pd.DataFrame(out).to_markdown())
    if "claims" in t:
        c = t["claims"].copy()
        ld, rd = pd.to_datetime(c["loss_date"]), pd.to_datetime(c["report_date"])
        r = {"sinistri": len(c), "stati": c["status"].value_counts().to_dict(),
             "quota respinti": round(float((c["status"].astype(str).str.lower().isin(["repudiated", "declined", "rejected"])).mean()), 4),
             "duplicati esatti": int(c.duplicated(subset=[x for x in ["policy_id", "loss_date", "peril", "incurred_to_date", "report_date"] if x in c.columns]).sum()),
             "lag denuncia mediano / 90% (gg)": f"{(rd - ld).dt.days.median():.0f} / {(rd - ld).dt.days.quantile(.9):.0f}",
             "lag negativo": int(((rd - ld).dt.days < 0).sum()),
             "incurred ≤ 0": int((c["incurred_to_date"] <= 0).sum()) if "incurred_to_date" in c else None,
             "recoveries > incurred": int((c["recoveries"] > c["incurred_to_date"]).sum()) if "recoveries" in c else None,
             "anni di accadimento": f"{ld.dt.year.min()}–{ld.dt.year.max()}",
             "perili per linea": {l: sorted(c.loc[c["line"] == l, "peril"].dropna().unique().tolist()) for l in c["line"].dropna().unique()}}
        # claims without an active policy at the loss date
        pol = pd.concat([t[f"policies_{l}"][["policy_id", "period_start", "period_end"]] for l in lines if f"policies_{l}" in t])
        m = c[["claim_id", "policy_id", "loss_date"]].merge(pol, on="policy_id", how="left")
        ok = (pd.to_datetime(m["loss_date"]) >= pd.to_datetime(m["period_start"])) & (pd.to_datetime(m["loss_date"]) <= pd.to_datetime(m["period_end"]))
        r["sinistri senza riga di polizza (quota)"] = round(1 - ok.groupby(m["claim_id"]).any().mean(), 4)
        md.append("\n**Sinistri**\n")
        md.append("\n".join(f"- {k}: {v}" for k, v in r.items()))
    md.append("")


def levels(t: dict, lines, md: list):
    """frequency, severity, burning cost by line and peril (reported claims, not developed)"""
    if "claims" not in t:
        return None
    c = t["claims"].copy()
    c = c[~c["status"].astype(str).str.lower().isin(["repudiated", "declined", "rejected"])]
    c["net"] = c["incurred_to_date"] - c.get("recoveries", 0)
    c["ay"] = pd.to_datetime(c["loss_date"]).dt.year
    rows = []
    for line in lines:
        k = f"policies_{line}"
        if k not in t:
            continue
        p = t[k]; p_year = pd.to_datetime(p["period_start"]).dt.year
        ex = p.groupby(p_year)["exposure"].sum()
        cl = c[c["line"] == line]
        for per, g in cl.groupby("peril"):
            n = g.groupby("ay").size().reindex(ex.index).fillna(0); cost = g.groupby("ay")["net"].sum().reindex(ex.index).fillna(0)
            rows.append({"linea": line, "perilo": per, "freq": float(n.sum() / ex.sum()), "sev": float(cost.sum() / max(n.sum(), 1)),
                         "burning cost": float(cost.sum() / ex.sum()), "quota aperti": float(g["status"].astype(str).str.lower().eq("open").mean()),
                         "n sinistri": int(n.sum()), "freq per anno": {int(y): round(float(v), 4) for y, v in (n / ex).items()}})
        n = cl.groupby("ay").size().reindex(ex.index).fillna(0); cost = cl.groupby("ay")["net"].sum().reindex(ex.index).fillna(0)
        rows.append({"linea": line, "perilo": "TOTALE", "freq": float(n.sum() / ex.sum()), "sev": float(cost.sum() / max(n.sum(), 1)),
                     "burning cost": float(cost.sum() / ex.sum()), "quota aperti": float(cl["status"].astype(str).str.lower().eq("open").mean()),
                     "n sinistri": int(n.sum()), "freq per anno": {int(y): round(float(v), 4) for y, v in (n / ex).items()}})
    return pd.DataFrame(rows)


def main(raw_new, raw_old, out_md, lines):
    new, old = load_tables(Path(raw_new), lines), load_tables(Path(raw_old), lines)
    md = [f"# Audit mercato nuovo — `{raw_new}` contro Parte 1 `{raw_old}`\n",
          "Generato da `src/audit_new_market.py`. Le soglie PSI sono indicative; ogni voce va letta, non applicata alla cieca.\n"]
    compare_schema(new, old, md)
    cols = compare_columns(new, old, md)
    quality_checks(new, lines, md, "mercato nuovo")
    ln, lo = levels(new, lines, md), levels(old, lines, md)
    md.append("## 4. Livelli tecnici per linea e perilo (sinistri denunciati, non sviluppati)\n")
    if ln is not None and lo is not None:
        j = ln.merge(lo, on=["linea", "perilo"], how="outer", suffixes=(" nuovo", " P1"))
        for c in ["freq", "sev", "burning cost"]:
            j[f"{c} nuovo/P1"] = (j[f"{c} nuovo"] / j[f"{c} P1"]).round(3)
        show = j[["linea", "perilo", "n sinistri nuovo", "freq nuovo", "freq P1", "freq nuovo/P1", "sev nuovo", "sev P1", "sev nuovo/P1",
                  "burning cost nuovo", "burning cost P1", "burning cost nuovo/P1", "quota aperti nuovo", "quota aperti P1"]].copy()
        for c in show.columns:
            if show[c].dtype == float:
                show[c] = show[c].round(4)
        md.append(show.to_markdown(index=False))
        md.append("\n**Frequenza per anno di accadimento (mercato nuovo)** — serve a vedere trend e maturità:\n")
        md.append(ln[["linea", "perilo", "freq per anno"]].to_markdown(index=False))
    elif ln is not None:
        md.append(ln.drop(columns=["freq per anno"]).round(4).to_markdown(index=False))
    # automatic summary
    md.append("\n## 5. Sintesi automatica (da verificare a mano)\n")
    def names(df):
        g = df.groupby("colonna")["tabella"].apply(lambda t: ", ".join(sorted(set(x.split("_")[0] for x in t))))
        return "; ".join(f"{c} ({t})" for c, t in g.head(15).items())
    big = cols[cols["PSI"] > 0.25]; mid = cols[(cols["PSI"] > 0.10) & (cols["PSI"] <= 0.25)]
    md.append(f"- Variabili con distribuzione molto diversa (PSI > 0,25): {big['colonna'].nunique()} → {names(big)}")
    md.append(f"- Variabili diverse (PSI 0,10–0,25): {mid['colonna'].nunique()} → {names(mid)}")
    nl = cols[cols["livelli nuovi"].astype(str).str.len() > 0]
    md.append(f"- Categoriche con livelli mai visti nella Parte 1: {nl['colonna'].nunique()} → {names(nl)}")
    missing = [k for k in old if k not in new]
    md.append(f"- Tabelle della Parte 1 assenti: {', '.join(missing) or 'nessuna'}")
    if ln is not None and lo is not None:
        tot = j[j["perilo"] == "TOTALE"]
        md.append("- Burning cost nuovo / Parte 1 per linea: " + ", ".join(f"{r.linea} {r['burning cost nuovo/P1']:.2f}" for _, r in tot.iterrows()))
    Path(out_md).write_text("\n".join(md))
    cols.to_csv(Path(out_md).with_suffix(".csv"), index=False)
    print(f"audit written to {out_md} ({len(cols)} columns compared)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lines = DEFAULT_LINES
    for a in sys.argv[1:]:
        if a.startswith("--lines"):
            lines = a.split("=")[1].split(",")
    main(args[0], args[1], args[2], lines)
