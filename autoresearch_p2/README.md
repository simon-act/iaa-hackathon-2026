# Loop Karpathy — Parte 2 (mercato emerging), 25/10

Qwen (locale, Ollama) modifica `experiment.py`; `eval_experiment.py` (congelato) lo valuta su 7 verifiche (2023→2024, 2024→2023,
5 gruppi di polizze) contro "prior del developed × livello"; si tiene solo se migliora sia il giudice senza spalmatura sia la forma
ufficiale. Dati: `data_p2.parquet` (costruito da `build_data.py`, NON nel repo). Avvio: `Avvia loop p2.command`; stop: `touch STOP`.
Risultati: `results.tsv`, `loop.log`, storia git della cartella. Nessun risultato del loop entra in una sub senza ri-verifica sul banco
completo (`scripts/p2_bench.py`).
