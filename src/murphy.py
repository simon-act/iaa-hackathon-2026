"""Scomposizione di Murphy del punteggio Tweedie (proper scoring rule, divergenza di Bregman).
S(pred) = UNC − DSC + MCB, con S = devianza media, ricalibrazione isotonica (PAV, ottimale per ogni Bregman):
  UNC = S(media)          incertezza del target (fissa)
  DSC = S(media) − S(iso) discriminazione: quanto il modello ordina bene i rischi
  MCB = S(pred) − S(iso)  errore di calibrazione: perso per livelli sbagliati a parità di ordinamento
  D² = (DSC − MCB) / UNC
Versione onesta: isotonica stimata su un fold e applicata all'altro.
Uso: python -I murphy.py <oof_name>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, str(Path(__file__).parent))
import prepare as P
ROOT = Path(__file__).resolve().parents[1]
dev = P.tweedie_deviance


def iso_fit(p, y):
    return IsotonicRegression(out_of_bounds="clip", increasing=True).fit(np.log(p), y)


def main(name):
    o = pd.read_parquet(ROOT / "results" / "oof" / f"{name}.parquet")
    rows = []
    for line in P.LINES + ["tutte"]:
        for f in ("F2023", "F2024"):
            s = o[(o.fold == f) & ((o.line == line) if line != "tutte" else True)]
            y, p = s.target_b.to_numpy(), s.pred.to_numpy()
            unc = dev(y, np.full_like(y, y.mean()))
            iso = iso_fit(p, y).predict(np.log(p))
            other = o[(o.fold != f) & ((o.line == line) if line != "tutte" else True)]
            iso_x = iso_fit(other.pred.to_numpy(), other.target_b.to_numpy()).predict(np.log(p))
            iso_x = np.clip(iso_x, 1e-3, None) * (p.sum() / np.clip(iso_x, 1e-3, None).sum())  # stesso livello totale
            sp, si, sx = dev(y, p), dev(y, np.clip(iso, 1e-3, None)), dev(y, iso_x)
            rows.append(dict(line=line, fold=f, D2=round(1 - sp / unc, 5), DSC=round((unc - si) / unc, 5),
                             MCB=round((sp - si) / unc, 5), D2_ricalib_onesta=round(1 - sx / unc, 5),
                             AE=round(y.sum() / p.sum(), 4)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (ROOT / "results" / "diag" / f"murphy_{name}.json").write_text(df.to_json(orient="records", indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
