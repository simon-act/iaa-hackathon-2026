"""22/10 sera: sub scelta TUTTA sul giudice ufficiale (target_b con IBNR e grandi sinistri spalmati per esposizione, come Kaggle).
Ipotesi: chi ci sta davanti in classifica (+0,001) guadagna avvicinandosi alla costruzione del target di Kaggle; noi abbiamo scelto la
forma sul giudice senza spalmatura. Qui, per linea, si sceglie su OOF 2023 e 2024 (criterio = Δ dell'anno peggiore sul giudice ufficiale):
  - componente freq×sev e componente GBM (solo quelle con OOF nei due fold);
  - peso del freq×sev w (0–1);
  - credibilità k (2…∞) applicata al blend, con o senza esclusione del cedimento nel fabbricato.
Nessun ritocco a posteriori: sono scelte di modello (componenti, peso, parametro di credibilità). Δ in e·4 rispetto a sub_41.
Uso: IAA_MARKET=configs/parte1_s3.json python3 scripts/official_opt.py [linea] ; output results_s3/official_opt.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS
ALL = {n: pd.read_parquet(R / f"oof/{n}.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"] for n in ("fs2_v2_cred", "qwen_gbm73_cred")}


def part(n, l, f):
    p = R / f"oof_parts/{n}_{l}_{f}.parquet"
    return pd.read_parquet(p).set_index("row_id")["pred"] if p.exists() else ALL[n]


# (freq×sev candidati, GBM candidati); il primo di ciascuno + w0/k0/excl0 = sub_41
CAND = {"motor": (["fs2_v2fb_cred", "fs2_v2f_cred", "fs2_v2fs_cred", "fs2_v2fb_llr_cred", "fs2_v2_cred"], ["qwen_gbm73fix_cred", "qwen_gbm73_cred"], 0.9, False),
        "buildings": (["fs2_v2_llr_cred", "fs2_v2_cred", "fs2_v2f_cred"], ["qwen_gbm73_gross_cred", "qwen_gbm73_cred"], 0.6, True),
        "contents": (["fs2_v2_sevmix_cred", "fs2_v2_cred", "fs2_v2_llr_cred"], ["qwen_gbm73_cred", "qwen_gbm73_gross_cred"], 0.8, False)}
KS = [2, 4, 8, 16, 32, 1e9]
WS = np.round(np.arange(0.0, 1.01, 0.1), 2)
ref = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "line", "fold", "exposure", "target_b"])
lines = sys.argv[1:] or list(CAND)
rows = []
for l in lines:
    fsl, gbl, w0, ex0 = CAND[l]
    for f in P.FOLDS:
        s = ref[(ref.line == l) & (ref.fold == f)].reset_index(drop=True); rid = s.row_id
        un = {n: X.uncred(part(n, l, f).reindex(rid).to_numpy(), s, l, 8) for n in fsl + gbl}

        def cred(p, k, ex):
            if k >= 1e9:
                return p
            return X.apply_excl(p, s, l, k) if ex else X.apply(p, s, l, k)
        p0 = cred(w0 * un[fsl[0]] + (1 - w0) * un[gbl[0]], 8, ex0)
        u0, o0, c0 = P.d2(s.target_b, p0), P.d2_own(rid, p0), P.d2_unc(rid, p0)
        for a in fsl:
            for b in gbl:
                for w in WS:
                    pb = w * un[a] + (1 - w) * un[b]
                    for ex in ([False, True] if l == "buildings" else [False]):
                        for k in KS:
                            if ex and k >= 1e9:
                                continue
                            p = cred(pb, k, ex)
                            rows.append(dict(line=l, fold=f, fs=a, gbm=b, w=w, k=k, excl=ex, uff=(P.d2(s.target_b, p) - u0) * 1e4,
                                             own=(P.d2_own(rid, p) - o0) * 1e4, unc=(P.d2_unc(rid, p) - c0) * 1e4, tot=p.sum() / p0.sum()))
        print(l, f, "fatto", len(rows), flush=True)
    r = pd.DataFrame(rows); out = R / "official_opt.csv"
    r.to_csv(out, mode="a", header=not out.exists(), index=False); rows = []
    t = r.pivot_table(index=["fs", "gbm", "w", "k", "excl"], columns="fold", values=["uff", "own", "unc"])
    t["min_uff"] = t["uff"].min(axis=1)
    print(l, "migliori sul giudice ufficiale (anno peggiore):")
    print(t.sort_values("min_uff", ascending=False).head(12).round(2).to_string(), flush=True)
