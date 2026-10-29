"""Summary of the bench validation (p2_bench32): for each comparison with a known public outcome, mean Δ e·4 and checks won by
judge: forma and own (current bench), forma_w and own_w (weighted to the 2025 mix), forma_nb (new business only)."""
import glob, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
r = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(P.RESULTS) / "p2_bench32_*_*.csv"))], ignore_index=True)
PAIRS = [("C1 sub_19 vs sub_18 (premio sulla frequenza)", "motor", "sub_20", "sub_18", "+0,00075"),
         ("C2 sub_20 vs sub_19 (prior P1 v2 in casa)", "contents", "sub_21", "sub_19", "+0,00036"),
         ("C2 sub_20 vs sub_19 (prior P1 v2 in casa)", "buildings", "sub_21", "sub_19", "+0,00036"),
         ("C3 sub_21 vs sub_20 (premio nel GLM del costo)", "motor", "sub_21", "sub_20", "+0,0009"),
         ("C4 offset Tweedie vs sub_21", "motor", "tw05", "sub_21", "+0,0012 (forma ~+0,0008)"),
         ("C4 offset Tweedie vs sub_21", "contents", "tw05", "sub_21", "+0,0012 (forma ~+0,0008)"),
         ("C5 fuoco vs sub_21", "buildings", "fuoco05", "sub_21", "+0,0007"),
         ("C6 stagionalità vs sub_21", "motor", "stag", "sub_21", "0,0000"),
         ("C6 stagionalità vs sub_21", "contents", "stag", "sub_21", "0,0000"),
         ("C6 stagionalità vs sub_21", "buildings", "stag", "sub_21", "0,0000")]
J = ["forma", "own", "forma_w", "own_w", "forma_nb"]
out = []
for name, li, a, b, pub in PAIRS:
    for bk in (7, 27):
        g = r[(r.linea == li) & (r.bank == bk)].set_index(["fold", "variante"])
        A, B = g.xs(a, level="variante"), g.xs(b, level="variante")
        d = (A[J] - B[J]) * 1e4; n = len(d)
        out.append(dict(confronto=name, linea=li, banco=bk, pubblico=pub,
                        **{j: f"{d[j].mean():+.1f} ({(d[j] > 0).sum()}/{n})" for j in J}))
s = pd.DataFrame(out); s.to_csv(Path(P.RESULTS) / "p2_bench32_sintesi.csv", index=False)
pd.set_option("display.width", 250); print(s.to_string(index=False))
