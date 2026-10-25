"""Variabili escluse per leakage/etica/proxy: lista esplicita con motivazione e test che nessun modello le usi (23/10).
Esecuzione: MODEL_FEATURES=results_s3/features_used.txt PYTHONPATH=src python3 -m pytest hackathon_checks/tests/test_excluded_features.py -q
MODEL_FEATURES: una variabile per riga, esportato da prepare.allowed_features (scripts/export_features.py)."""
import json, os
from pathlib import Path
import pytest

EXCLUDED = {
    "customer_service_contacts": "leakage: generati dai sinistri (anche non denunciati); nel test ha la distribuzione delle righe senza sinistri — usati solo come segnale di sinistri tardivi nella storia (experience.history_fix)",
    "renewal_decision": "leakage: decisa dopo il periodo, assente nel test",
    "written_premium": "circolare: tariffa attuale, solo benchmark",
    "row_id": "id", "policy_id": "id", "customer_id": "id (solo per la storia)", "address_id": "id (solo per la storia)", "claim_id": "id",
    "surname": "proxy etnia / nessuna logica di rischio", "email_domain": "nessuna logica di rischio",
    "date_of_birth": "sostituita dall'età", "birth_year": "sostituita dall'età", "construction_type_raw": "grezzo, sostituito dalla versione pulita",
}
FEATURES_FILE = os.environ.get("MODEL_FEATURES")


def _features():
    txt = Path(FEATURES_FILE).read_text()
    try:
        return set(json.loads(txt))
    except json.JSONDecodeError:
        return {l.strip() for l in txt.splitlines() if l.strip()}


@pytest.mark.skipif(not FEATURES_FILE, reason="impostare MODEL_FEATURES")
def test_no_excluded_feature_used():
    used = _features()
    bad = sorted(f for f in used if f.lower() in EXCLUDED)
    assert not bad, f"variabili escluse usate dal modello: {bad}"


def test_every_exclusion_has_reason():
    assert all(v.strip() for v in EXCLUDED.values())


def test_consistent_with_pipeline_lists():
    try:
        import prepare  # noqa: F401
    except Exception:
        pytest.skip("pipeline non importabile (PYTHONPATH=src)")
    forbidden = {str(x).lower() for x in getattr(prepare, "FORBIDDEN", [])}
    missing = {"customer_service_contacts", "renewal_decision", "written_premium"} - forbidden
    assert not missing, f"mancano in prepare.FORBIDDEN: {missing}"
