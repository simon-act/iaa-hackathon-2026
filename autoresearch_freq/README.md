# Loop Qwen sulla FREQUENZA per perilo (20/10)
- `data/`: dataset compatto per linea e anno (s3, holdout escluso) con variabili, n_<perilo>, pi_<perilo>; `meta.json` con perili, variabili, trend, quote di costo.
- `eval_experiment.py`: giudice congelato, D² Poisson dei conteggi per perilo (forma), fold peggiore. Riproduce esattamente scripts/freq_bench.py del container.
- `experiment.py`: partenza = frequenza attuale del freq×sev.
- Avvio: doppio clic su `Avvia loop frequenza.command` (ferma gli altri loop). Stop: `touch autoresearch_freq/STOP`.
- Regola: ogni idea tenuta viene rifatta da Claude nel modello completo (OOF con i tre giudici) prima di essere usata.
