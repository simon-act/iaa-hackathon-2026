"""Parte 2: premio commerciale (office premium) delle 20 polizze di riferimento dal premio di rischio previsto.

Formula (ipotesi dichiarate nel report):
    premio = (premio di rischio + spesa fissa × esposizione + riass. cat × premio di rischio [solo casa e contenuto])
             / (1 − spese variabili − commissione del canale − imposta − margine di profitto)
- spesa fissa "per polizza" riportata all'esposizione della riga (una riga di 0,1 anni non paga una spesa annua intera);
- riassicurazione catastrofale = quota del premio di rischio delle linee property (fabbricato e contenuto), 0 in auto;
- commissione per canale; nel mercato emerging il comparatore non ha un valore (vuoto) → 0, segnalato;
- imposta e profitto come quote del premio commerciale.
Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_office_premium.py submissions_p2/<file>.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402

RAW = Path(P.CFG["raw"])
ex = pd.read_parquet(RAW / "expenses.parquet").set_index("item")
ref = pd.read_parquet(RAW / "reference_policies.parquet")
pred = pd.read_csv(sys.argv[1]).set_index("row_id")["prediction"]


def build(row, mkt="emerging"):
    v = lambda k: float(ex.loc[k, mkt]) if pd.notna(ex.loc[k, mkt]) else 0.0
    rp = float(pred.get(row.row_id, np.nan))
    fixed = v("fixed_expense_per_policy") * row.exposure
    cat = v("cat_reinsurance") * rp if row.line in ("buildings", "contents") else 0.0
    comm = v(f"commission_{row.channel}")
    load = v("variable_expense") + comm + v("premium_tax") + v("target_profit_margin")
    op = (rp + fixed + cat) / (1 - load)
    return dict(row_id=row.row_id, line=row.line, channel=row.channel, exposure=row.exposure, risk_premium=round(rp, 2),
                fixed_expense=round(fixed, 2), cat_reins=round(cat, 2), loadings_share=round(load, 3),
                office_premium=round(op, 2), written_premium=row.written_premium, ratio_to_written=round(op / row.written_premium, 2))


out = pd.DataFrame([build(r) for r in ref.itertuples()])
pd.set_option("display.width", 220)
print(out.to_string(index=False))
dst = P.RESULTS / "office_premium_reference.csv"; out.to_csv(dst, index=False); print("->", dst)
