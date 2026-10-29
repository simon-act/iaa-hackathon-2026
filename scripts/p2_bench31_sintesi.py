"""Summary of the D tests (p2_bench31): Δ e·4 vs sub_21 by bench, with the rules fixed in advance."""
import glob, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
r = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(P.RESULTS) / "p2_bench31_*_*_*.csv"))])
r["linea"] = [Path(f).name for f in [""] * len(r)]
out = []
for f in glob.glob(str(Path(P.RESULTS) / "p2_bench31_*_*_*.csv")):
    d = pd.read_csv(f); d["linea"] = Path(f).name.split("_")[2]; out.append(d)
r = pd.concat(out, ignore_index=True)
base = r[r.variante == "sub_21"].set_index(["linea", "bank", "fold"])
r = r[r.variante != "sub_21"].copy()
for c in ("forma", "own"):
    r["d_" + c] = (r[c].to_numpy() - base.loc[list(zip(r.linea, r.bank, r.fold)), c].to_numpy()) * 1e4
rows = []
for (li, bk, v), g in r.groupby(["linea", "bank", "variante"]):
    n = len(g); wf, wo = (g.d_forma > 0).sum(), (g.d_own > 0).sum()
    if bk == 7:
        ok = wf >= 5 and wo >= 5
        dirs = ""
    else:
        dm = g.assign(dr=g.fold.str[:9]).groupby("dr").d_forma.mean()
        ok = g.d_forma.mean() > 0 and g.d_own.mean() > 0 and wf >= 4 and wo >= 4 and (dm > 0).all()
        dirs = " / ".join(f"{k}: {v:+.1f}" for k, v in dm.items())
    rows.append(dict(linea=li, banco=bk, variante=v, forma=round(g.d_forma.mean(), 1), own=round(g.d_own.mean(), 1),
                     vinte=f"{wf}/{n} · {wo}/{n}", direzioni_forma=dirs, passa="sì" if ok else "no"))
s = pd.DataFrame(rows).sort_values(["linea", "variante", "banco"])
s.to_csv(Path(P.RESULTS) / "p2_bench31_sintesi.csv", index=False)
print(s.to_string(index=False))
