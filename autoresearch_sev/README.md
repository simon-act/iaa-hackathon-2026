# Loop sulla severità (dal 19/10 pomeriggio)

Terzo loop. Il primo riscriveva un GBM da zero (316 iterazioni, 9 tenute, nulla dopo la 211); il secondo cercava correzioni sui residui del modello migliore (62 iterazioni, 1 tenuta: valore del veicolo; conferma che le relatività sono giuste).
Questo isola la metà del modello dove c'è più spazio e meno dati: la severità per perilo. `claims_sev.parquet` = un record per sinistro con le variabili della riga di polizza, perilo, anno, costo (tagliato, sviluppato, al livello 2025).
Giudice: D² Gamma sull'anno dopo (fit ≤ 2022 → 2023, fit ≤ 2023 → 2024), perili pesati per costo, linee come Kaggle; punteggio = fold peggiore. Base = il modello attuale: 0,02685 / 0,0342. Un giro dura pochi secondi.
Avvio: doppio clic su `Avvia loop severita.command` (ferma gli altri loop). Stop: `touch autoresearch_sev/STOP`.
