"""Contract tests on the submission (pytest), 23/10.
Run: SUBMISSION=submissions/sub_50_contatti_storia.csv SAMPLE=data/raw/public/sample_submission.parquet \
  EXPECTED_LEVELS='{"M": 567.8, "B": 517.6, "C": 130.7}' TEST_FILES=data/raw/public/test_motor.parquet,... python3 -m pytest hackathon_checks/tests -q
EXPECTED_LEVELS = expected cost per exposure-year by line (tolerance LEVEL_TOL, default 0.06)."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, pytest

SUB = os.environ.get("SUBMISSION"); SAMPLE = os.environ.get("SAMPLE")
pytestmark = pytest.mark.skipif(not SUB or not SAMPLE, reason="impostare SUBMISSION e SAMPLE")


def _load(p):
    p = Path(p); return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)


@pytest.fixture(scope="module")
def sub():
    s = _load(SUB); s["row_id"] = s["row_id"].astype(str); return s


@pytest.fixture(scope="module")
def sample():
    s = _load(SAMPLE); s["row_id"] = s["row_id"].astype(str); return s


def test_columns(sub):
    assert list(sub.columns) == ["row_id", "prediction"], sub.columns.tolist()


def test_row_ids_match_sample(sub, sample):
    assert len(sub) == len(sample) and sub.row_id.is_unique and set(sub.row_id) == set(sample.row_id)


def test_predictions_finite_positive(sub):
    p = sub.prediction.astype(float)
    assert np.isfinite(p).all() and (p > 0).all(), "Tweedie 1<p<2: never 0 or negative"


def test_no_extreme_outliers(sub):
    p = sub.prediction.astype(float); line = sub.row_id.str[:1]
    for l, g in p.groupby(line):
        med = g.median()
        assert g.max() < 200 * med, f"line {l}: max {g.max():.0f} vs median {med:.1f} (row with SI x1000?)"
        assert g.quantile(0.999) < 30 * med, f"line {l}: p99.9 {g.quantile(0.999):.0f} vs median {med:.1f}"


def test_levels_per_line(sub):
    lv = os.environ.get("EXPECTED_LEVELS"); tf = os.environ.get("TEST_FILES")
    if not lv or not tf:
        pytest.skip("EXPECTED_LEVELS / TEST_FILES not set")
    levels = json.loads(Path(lv).read_text()) if Path(lv).exists() else json.loads(lv)
    tol = float(os.environ.get("LEVEL_TOL", 0.06)); expo = []
    for f in tf.split(","):
        d = _load(f)
        s = pd.to_datetime(d["period_start"]); e = pd.to_datetime(d["period_end"])
        expo.append(pd.DataFrame({"row_id": d["row_id"].astype(str), "expo": ((e - s).dt.days + 1) / 365.0}))
    expo = pd.concat(expo).set_index("row_id")["expo"]
    m = sub.set_index("row_id").join(expo)
    for l, g in m.groupby(m.index.str[:1]):
        if l in levels:
            lvl = g.prediction.sum() / g.expo.sum()
            assert abs(lvl / levels[l] - 1) < tol, f"line {l}: level {lvl:.1f} vs expected {levels[l]} (tol {tol:.0%})"
