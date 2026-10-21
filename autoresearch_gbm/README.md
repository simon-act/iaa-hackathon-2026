# Loop Qwen sul GBM (20/10) — giudice senza spalmatura, due fold, target gross
- `data/`: dataset compatto per anno e linea (s3, holdout escluso), con le colonne del giudice (target_b, target_gross, target_rep, pi_eff, ll_expo).
- `eval_experiment.py`: giudice congelato. D² senza spalmatura (forma), punteggio = fold peggiore; stampa anche il giudice ufficiale.
- `experiment.py`: partenza = GBM di Qwen it. 73 corretto da Claude (senza vincolo monotono su vehicle_age), stimato sul target gross.
- `program.md`: istruzioni e cose verificate dal team.
- Avvio: doppio clic su `Avvia loop GBM.command`. Stop: `touch autoresearch_gbm/STOP`.
- Regola: ogni idea tenuta viene rifatta da Claude nel container con i tre giudici (ufficiale, senza spalmatura, con grandi sinistri veri) prima di entrare nel blend.
