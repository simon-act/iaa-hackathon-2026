# Loop sulla storia della polizza (dal 21/10 sera)
Il pezzo grande che il modello non spiega è il rischio proprio della polizza (D² ≈ 0,043 non recuperabile, docs/22 §7);
la storia ne recupera solo 0,002–0,004. Il loop cerca un uso migliore della storia (credibilità) sopra il blend di sub_41.
- `base_oof.parquet`: OOF 2023/2024 del blend di sub_41 (`scripts/base41_oof.py`), con e senza credibilità, + colonne del giudice.
- `storia.parquet`: storia per riga (sinistri passati per perilo, anno prima, cliente sulle altre linee, indirizzo) — `scripts/build_storia.py`.
- `eval_experiment.py`: giudice congelato (senza spalmatura, incrociato tra anni, verso peggiore; riporta anche l'ufficiale).
- Avvio: `Avvia loop storia.command` oppure job del runner. Stop: `touch autoresearch_storia/STOP`.
- Regola: ogni idea tenuta viene rifatta da Claude come modifica del modello (credibilità in src/experience.py) e giudicata sui tre giudici prima di entrare in una sub.
