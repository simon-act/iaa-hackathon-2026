# Loop sui residui (dal 19/10)

Il loop vecchio (`autoresearch/`) riscriveva da zero un GBM: dopo 285 iterazioni ripeteva le stesse idee (9 tenute, nessuna dopo la 211).
Questo loop parte dal modello migliore (blend di sub_18, previsioni fuori campione 2023 e 2024 in `base_oof.parquet`) e cerca
quello che sbaglia ancora, in modo che regga da un anno all'altro: si stima su un anno e si valuta sull'altro, nei due versi;
il punteggio è il guadagno di D² nel verso peggiore.

- `eval_experiment.py`: giudice congelato. `experiment.py`: il file che Qwen modifica. `program.md`: istruzioni e cosa sappiamo.
- Partenza: calibrazione a potenza per linea (pred^b), +0,00038 nel verso peggiore.
- Già provato: un GBM di correzione su tutte le variabili peggiora (−0,0079 con 150 giri), quindi i residui sono quasi tutto rumore.
- Avvio: doppio clic su `Avvia loop residui.command` (ferma prima il loop vecchio). Stop: `touch autoresearch_residui/STOP`.
