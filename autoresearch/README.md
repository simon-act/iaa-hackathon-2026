# Loop autoresearch (stile Karpathy) con Qwen locale

Un LLM locale (Qwen via Ollama) propone **una modifica alla volta** a `experiment.py`. Un giudice congelato la misura sui fold temporali; la modifica resta solo se migliora il punteggio.

| File | Chi lo tocca | Ruolo |
|---|---|---|
| `program.md` | solo Simone | direzione di ricerca, regole, idee da provare |
| `experiment.py` | solo il loop | l'unico file che l'LLM può cambiare (parte dal GBM del passo 2) |
| `eval_experiment.py` + `../src/prepare.py` | nessuno | giudice: fold 2023/2024, holdout chiuso, Tweedie D² p=1,5 |
| `run_loop.py` | nessuno | ciclo: chiede, controlla, valuta, tiene o scarta, registra |
| `results.tsv` | il loop | una riga per tentativo (idea, punteggio, esito, tempo) |
| `runs/iter_XXXX.py` | il loop | ogni proposta, anche quelle scartate |
| `.git` (in questa cartella) | il loop | un commit per ogni modifica tenuta |

## Guardrail

- L'esperimento riceve solo colonne ammesse; il target c'è solo per le righe di stima, mai per quelle di validazione; l'holdout (10% polizze) non viene mai passato.
- Controllo statico prima di eseguire: solo import ammessi, niente file, rete o processi, firma di `fit_predict` invariata.
- Tempo massimo per prova (`--timeout`, default 900 s); previsioni finite e ≥ 0.
- Si tiene una modifica solo se migliora `pooled_mean` di almeno 0,0003 (filtro contro il rumore).
- Alla fine si controlla l'holdout chiuso (winner's curse): migliorare sui fold non basta.

## Uso (sul Mac)

```bash
cd <repository>
bash autoresearch/setup_mac.sh      # una volta: .venv, librerie, dati, controllo Ollama
bash autoresearch/run_loop.sh       # avvia in background, Mac tenuto sveglio
tail -f autoresearch/loop.log       # seguire
touch autoresearch/STOP             # fermare
```

Opzioni: `--model qwen3.6` (default: il miglior Qwen installato), `--iters 200`, `--timeout 900`, `--min-delta 0.0003`.
Test senza LLM: `.venv/bin/python -I autoresearch/run_loop.py --llm mock --iters 3`.
