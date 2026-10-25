# Verifica delle segnalazioni di Qwen (19/10, 23:45) — Claude

La revisione è durata 79 minuti e ha prodotto 125 segnalazioni. Nella seconda passata Qwen ne ha smontate da solo 93 (FALSE_POSITIVE). Ne restano 3 reali e 8 incerte; 21 non sono state ricontrollate, quasi tutte di gravità bassa. Ognuna delle 11 è stata controllata a mano nel container.

| # | Segnalazione | Verifica | Azione |
|---|---|---|---|
| 1 | develop.py:71 — linee in una tupla fissa invece di `LINES` | **Reale** (Parte 2 con linee diverse) | Corretto: `for line in LINES`. Parte 1 identica |
| 2 | develop.py:86 — AY maturi `<= 2022` fisso | **Reale**: con un mercato 2023–25 nessun AY sarebbe maturo | Corretto: `<= YEARS[-1] − 2`. Parte 1 (2022) e mercato finto (2022) identici |
| 3 | experience.py:28 — `sev` della credibilità stimata su tutti gli anni, compresi quelli valutati nei fold | **Reale ma trascurabile**: è uno scalare per linea; con i soli anni di stima cambia dello 0,5–2% (auto 4.181 contro 4.087/4.125), un effetto sul D² molto più piccolo di k 6→12 (≤ 0,0002) | Documentato, non toccato |
| 4–5 | clean_claims.py:92 — colonna `recoveries` data per scontata | Incerta → **robustezza Parte 2** | Se manca: recuperi = 0, con avviso |
| 6–8 | features.py:135–138 — `birth_year`, `driver_age`, `years_licensed`, `annual_mileage` dati per scontati | Incerta → **robustezza Parte 2** | Colonne derivate calcolate solo se le sorgenti esistono |
| 9 | build_datasets.py:115 — `credit_band` nel report | Solo il report di controllo: le rinomine della config passano prima | Nessuna |
| 10 | models_extra.py:44 — `SEV_DRIVERS` fissi | Il codice filtra già con `if c in base_cols` | Nessuna |
| 11 | holdout_weights.py:27 — `row_id` dato per scontato | `row_id` è creato dalla pipeline (clean_policies), non viene dal mercato | Nessuna |

Bilancio: nessun leakage rilevante e nessun bug che cambi i numeri della Parte 1. Restano 4 fragilità vere per la Parte 2, ora corrette. Il tasso di falsi positivi è alto (93 su 104 high/medium), ma la seconda passata li filtra bene: delle 11 rimaste, 7 erano punti veri, anche se piccoli.
