# Revisione del codice con Qwen (19/10 sera)

Quarto uso di Qwen, diverso dai loop: invece di proporre modelli, **legge la pipeline e cerca errori**.
- `code_snapshot/`: copia dei file della pipeline usati per la sub finale (src/ del container al 19/10, commit 5b4efe1 + s4).
- `CONTEXT.md`: cosa sa il revisore del progetto (gara, tabelle, pipeline, fold, cosa conta come leakage).
- `review.py`: 25 pezzi di codice × 3 lenti (leakage, bug, robustezza Parte 2) = 75 chiamate; poi una seconda
  passata in cui Qwen prova a **smontare** ogni segnalazione high/medium (REAL / FALSE_POSITIVE / UNSURE).
- Output: `findings.jsonl`, `SINTESI.md` (tabella ordinata), `review.log`, `reviews/unparsed.txt`.
- Avvio: doppio clic su `Avvia revisione Qwen.command` (nella cartella del progetto). Stop: `touch autoresearch_review/STOP`.
  Riprende da dove era rimasto (`done.json`).
- Regola: nessuna segnalazione si applica senza verifica nel container (test riproducibile). Le verifiche vanno in
  `VERIFICA.md`. È anche materiale per il punteggio "trust/validation" e "uso dell'AI" (revisore indipendente locale).
