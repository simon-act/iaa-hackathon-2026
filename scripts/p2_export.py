"""Parte 2: export del prior dal mercato di partenza per le linee indicate (wrapper di src/transfer.py export-source).
Uso: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export.py configs/parte2.json buildings,contents,motor"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import transfer as T, prepare as P  # noqa: E402
P.LINES[:] = sys.argv[2].split(",")
T.export_source(sys.argv[1])
