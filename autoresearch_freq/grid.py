"""Per-peril frequency grid, run on the Mac (20/10). Same judge and same data as the loop (eval_experiment.py):
for each configuration and for each line, peril and fold, Poisson D² of the counts (shape only).
The configurations change the parameters of experiment_base.py (frozen copy of the current freq×sev frequency model).
Writes grid_results.csv row by row and resumes where it stopped. Usage: .venv/bin/python -I autoresearch_freq/grid.py
(config names: giri = boosting rounds, prof = depth; CSV columns: linea = line, perilo = peril, quota_costo = cost share)."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_experiment import META, pd2  # noqa: E402

CONFIGS = {
    "base": {}, "giri_x0.5": {"_mult": 2}, "giri_x2": {"_mult": 8}, "l2_1": {"lambda_l2": 1.0}, "l2_50": {"lambda_l2": 50.0},
    "minleaf_300": {"min_data_in_leaf": 300}, "minleaf_3000": {"min_data_in_leaf": 3000},
    "lr0.05_giri_x2": {"learning_rate": 0.05, "_mult": 8}, "prof2": {"num_leaves": 3, "max_depth": 2},
    "prof3": {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.05}, "ff0.7": {"feature_fraction": 0.7}, "ff0.5": {"feature_fraction": 0.5},
    "nobag": {"bagging_fraction": 1.0, "bagging_freq": 0}, "path_smooth50": {"path_smooth": 50.0}, "maxbin255": {"max_bin": 255},
    "ff0.7_maxbin255": {"feature_fraction": 0.7, "max_bin": 255}, "ff0.7_maxbin255_giri_x2": {"feature_fraction": 0.7, "max_bin": 255, "_mult": 8},
    "ff0.5_maxbin255": {"feature_fraction": 0.5, "max_bin": 255},
    "tweedie1.1": {"objective": "tweedie", "tweedie_variance_power": 1.1}, "tweedie1.3": {"objective": "tweedie", "tweedie_variance_power": 1.3},
    "extra_trees": {"extra_trees": True}, "maxbin255_l2_50": {"max_bin": 255, "lambda_l2": 50.0},
}
OUT = HERE / "grid_results.csv"


def load_base():
    spec = importlib.util.spec_from_file_location("exp_base", HERE / "experiment_base.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    done = set(map(tuple, pd.read_csv(OUT)[["config", "linea", "perilo", "fold"]].values)) if OUT.exists() else set()
    E = load_base(); P0 = dict(E.PARAMS); M0 = E.MULT
    for line in ("motor", "buildings", "contents"):
        meta = META[line]; feats = meta["features"]
        d = {y: pd.read_parquet(HERE / "data" / f"{line}_{y}.parquet") for y in (2021, 2022, 2023, 2024)}
        for last in (2022, 2023):
            tr = pd.concat([d[y] for y in range(2021, last + 1)], ignore_index=True); va = d[last + 1]
            for p in meta["perils"]:
                lt = np.log1p(meta["freq_trend"][p])
                off = lambda x: np.log(x["exposure"].to_numpy() * np.clip(x[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - x["year"].to_numpy()) * lt
                train = tr[feats + ["exposure", "year"]].assign(offset=off(tr), n=tr[f"n_{p}"].to_numpy(float))
                valid = va[feats + ["exposure", "year"]].assign(offset=off(va)); y = va[f"n_{p}"].to_numpy(float)
                for cname, cfg in CONFIGS.items():
                    key = (cname, line, p, f"F{last + 1}")
                    if key in done:
                        continue
                    cfg = dict(cfg); E.MULT = cfg.pop("_mult", M0); E.PARAMS = dict(P0, **cfg); t0 = time.time()
                    mu = np.asarray(E.fit_predict(train, valid, line, p), float); mu = mu * y.sum() / mu.sum()
                    row = {"config": cname, "linea": line, "perilo": p, "fold": f"F{last + 1}", "d2": pd2(y, mu),
                           "quota_costo": meta["cost_share"][p], "sec": round(time.time() - t0, 1)}
                    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                    print(row, flush=True)
    print("GRID_FINISHED", flush=True)


if __name__ == "__main__":
    main()
