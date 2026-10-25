"""Esporta le variabili ammesse dai modelli (prepare.allowed_features) per linea → results_s3/features_used.txt"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P
used = set()
for l in P.LINES:
    used |= set(P.allowed_features(P.load(l)))
(P.RESULTS / "features_used.txt").write_text("\n".join(sorted(used)) + "\n"); print(len(used), "variabili")
