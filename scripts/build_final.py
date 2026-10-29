"""Final submission (sub_40) rebuilt ONLY from the final fits in results_s3/final_parts (no intermediate files from sub_18).
Per-line blend: w × freq×sev + (1 − w) × GBM; for buildings the credibility excludes subsidence (apply_excl).
Before running it: the 6 components via scripts/final_model_line.py (see run_final.sh).
Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/build_final.py [file_name] [--check submissions/sub_40_finale_proposta.csv]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, experience as X  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]

# line: (freq×sev, GBM, freq×sev weight, credibility excluding subsidence)
SPEC = {"motor": ("fs2_v2fb_cred", "qwen_gbm73fix_gross_cred", 0.9, False),
        "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_cred", 0.6, True),
        "contents": ("fs2_v2_cred", "qwen_gbm73_cred", 0.8, False)}
# variants (--variante name): only the lines that differ from SPEC
VARIANTI = {"sub_41": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False)},   # 21/10: contents flood GBM×GLM
            # 22/10: motor large-loss load allocated by predicted risk (llr) or by bodily-injury frequency (llrbi), on top of sub_41
            "sub_45": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "motor": ("fs2_v2fb_llr_cred", "qwen_gbm73fix_gross_cred", 0.9, False)},
            "sub_46": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "motor": ("fs2_v2fb_llrbi_cred", "qwen_gbm73fix_gross_cred", 0.9, False)},
            # 22/10 evening: everything chosen on the official judge (scripts/official_opt.py): only motor credibility changes, k=8 → 16
            "sub_47": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False)},
            # 22/10 night: deductible relative to insured value (xs_rel) — buildings GBM (+ motor own-damage/theft frequency in sub_48)
            "sub_48": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True),
                       "motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_gross_cred", 0.9, False)},
            "sub_49": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True)},
            # 22/10 night: sub_48 + 2024 motor history completed with the late claims expected from contacts (experience.apply_fix)
            "sub_50": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True),
                       "motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_gross_cred", 0.9, "fix")},
            # 23/10 night: sub_50 + motor credibility k=16 (official judge and holdout in favour; own (no spreading) against)
            "sub_51": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True),
                       "motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_gross_cred", 0.9, "fix")},
            # 23/10 night: sub_50 + construction_type T and W kept separate in buildings (pipeline parte1_s3w, parts copied with suffix _w)
            "sub_52": {"contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False), "buildings": ("fs2_v2_llr_cred_w", "qwen_gbm73_gross_xs_cred_w", 0.6, True),
                       "motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_gross_cred", 0.9, "fix")}}
# credibility other than k=8 (blend without credibility, then credibility k on the blend), by variant and line
K_VAR = {"sub_47": {"motor": 16}, "sub_51": {"motor": 16}}


def part(name, line):
    f = P.RESULTS / "final_parts" / f"{name}_{line}.parquet"
    return pd.read_parquet(f).set_index("row_id")[f"p_{name}"]


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    name = args[0] if args else "sub_finale"
    out = []
    var = sys.argv[sys.argv.index("--variante") + 1] if "--variante" in sys.argv else None
    spec = dict(SPEC, **VARIANTI[var]) if var else SPEC
    for line, (a, b, w, ex) in spec.items():
        te = P.load(line, "test")[["row_id", "exposure"]]
        pa, pb = part(a, line).reindex(te.row_id).to_numpy(), part(b, line).reindex(te.row_id).to_numpy()
        if ex == "fix":   # last-year history completed (contacts)
            pa = X.apply_fix(X.uncred(pa, te, line, 8), te, line, 8)
            pb = X.apply_fix(X.uncred(pb, te, line, 8), te, line, 8)
            print(line, "expected late claims added", round(X.history_fix(line)[3], 1), "ΔH", round(X.history_fix(line)[4], 1), flush=True)
        elif ex:
            pa = X.apply_excl(X.uncred(pa, te, line, 8), te, line, 8)
            pb = X.apply_excl(X.uncred(pb, te, line, 8), te, line, 8)
        p = w * pa + (1 - w) * pb
        kv = K_VAR.get(var, {}).get(line)
        if kv:
            base = w * X.uncred(pa, te, line, 8) + (1 - w) * X.uncred(pb, te, line, 8)
            p = X.apply_fix(base, te, line, kv) if ex == "fix" else X.apply(base, te, line, kv)
        print(line, "rate", round(p.sum() / te.exposure.sum(), 1), flush=True)
        out.append(pd.DataFrame({"row_id": te.row_id.to_numpy(), "prediction": p}))
    o = pd.read_parquet(Path(P.CFG["raw"]) / "sample_submission.parquet", columns=["row_id"]).merge(pd.concat(out), on="row_id", how="left") \
        if (Path(P.CFG["raw"]) / "sample_submission.parquet").exists() else pd.concat(out)
    o["prediction"] = o.prediction.round(3)
    assert o.prediction.notna().all() and (o.prediction >= 0).all()
    o.to_csv(ROOT / "submissions" / f"{name}.csv", index=False)
    if "--check" in sys.argv:
        ref = pd.read_csv(ROOT / sys.argv[sys.argv.index("--check") + 1]).set_index("row_id").prediction.reindex(o.row_id)
        d = np.abs(o.prediction.to_numpy() - ref.to_numpy())
        print("comparison with", sys.argv[sys.argv.index("--check") + 1], "max |Δ|", float(d.max()), "rows with Δ>0.01:", int((d > 0.01).sum()),
              "ratio of totals", float(o.prediction.sum() / ref.sum()))


if __name__ == "__main__":
    main()
