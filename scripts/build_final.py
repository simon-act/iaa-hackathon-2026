"""Submission finale (sub_40) ricostruita SOLO dalle stime finali in results_s3/final_parts (nessun file intermedio di sub_18).
Blend per linea: w × freq×sev + (1 − w) × GBM; nel fabbricato la credibilità esclude il cedimento (apply_excl).
Prima di lanciarlo: le 6 componenti con scripts/final_model_line.py (vedi run_final.sh).
Uso: IAA_MARKET=configs/parte1_s3.json python3 scripts/build_final.py [nome_file] [--check submissions/sub_40_finale_proposta.csv]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]

# linea: (freq×sev, GBM, peso del freq×sev, credibilità senza cedimento)
SPEC = {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_gross_cred", 0.9, False),
        "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True),
        "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False)}
# varianti (--variante nome): solo le linee che cambiano rispetto a SPEC
VARIANTI = {"sub_41": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False)}}   # 21/10: alluvione contenuto GBM×GLM


def part(name, line):
    f = P.RESULTS / "final_parts" / f"{name}_{line}.parquet"
    return pd.read_parquet(f).set_index("row_id")[f"p_{name}"]


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    name = args[0] if args else "sub_finale"
    out = []
    spec = dict(SPEC, **VARIANTI[sys.argv[sys.argv.index("--variante") + 1]]) if "--variante" in sys.argv else SPEC
    for line, (a, b, w, ex) in spec.items():
        te = P.load(line, "test")[["row_id", "exposure"]]
        pa, pb = part(a, line).reindex(te.row_id).to_numpy(), part(b, line).reindex(te.row_id).to_numpy()
        if ex:
            pa = X.apply_excl(X.uncred(pa, te, line, 8), te, line, 8)
            pb = X.apply_excl(X.uncred(pb, te, line, 8), te, line, 8)
        p = w * pa + (1 - w) * pb
        print(line, "tasso", round(p.sum() / te.exposure.sum(), 1), flush=True)
        out.append(pd.DataFrame({"row_id": te.row_id.to_numpy(), "prediction": p}))
    o = pd.read_parquet(Path(P.CFG["raw"]) / "sample_submission.parquet", columns=["row_id"]).merge(pd.concat(out), on="row_id", how="left") \
        if (Path(P.CFG["raw"]) / "sample_submission.parquet").exists() else pd.concat(out)
    o["prediction"] = o.prediction.round(3)
    assert o.prediction.notna().all() and (o.prediction >= 0).all()
    o.to_csv(ROOT / "submissions" / f"{name}.csv", index=False)
    if "--check" in sys.argv:
        ref = pd.read_csv(ROOT / sys.argv[sys.argv.index("--check") + 1]).set_index("row_id").prediction.reindex(o.row_id)
        d = np.abs(o.prediction.to_numpy() - ref.to_numpy())
        print("confronto con", sys.argv[sys.argv.index("--check") + 1], "max |Δ|", float(d.max()), "righe con Δ>0,01:", int((d > 0.01).sum()),
              "rapporto totali", float(o.prediction.sum() / ref.sum()))


if __name__ == "__main__":
    main()
