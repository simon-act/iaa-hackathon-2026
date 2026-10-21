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

## Seconda verifica, indipendente da Qwen (20/10, 01:30) — Claude, su richiesta di Simone ("non mi fido di Qwen")
Le 93 segnalazioni che Qwen ha smontato da solo e le 21 senza seconda passata sono state rilette tutte. Quelle con un possibile effetto sui numeri sono state controllate sul codice e sui dati:

| # | Segnalazione | Verifica | Esito |
|---|---|---|---|
| 28 | trend.py:55 esposizione con `first` invece di `sum` | l'esposizione di linea×AY è la stessa su tutte le righe dei perili: sommarla la moltiplicherebbe per il numero di perili | corretto così |
| 27 | trend.py:126 stimatore di Pareto sbagliato | α = n / Σ log(x/u0) è l'MLE (Hill) con soglia nota; E[(X−cap)+] = λ·cap/(α−1) corretto | corretto (fragile solo con α ≤ 1 nella Parte 2) |
| 81 | GLM Poisson con previsione costante | `fit(A, n/esposizione, sample_weight=esposizione)` è la regressione Poisson dei tassi standard | corretto; glm_peril è debole per le variabili in classi, non per un errore |
| 78–80 | bin del GLM stimati anche sulla validazione | `_bin` stima i quantili solo sul train e li applica alla validazione | corretto |
| 91–92 | `_peril_hist` tiene il cumulato del primo AY | il ciclo va per AY crescente e sovrascrive: resta il cumulato all'ultimo AY ≤ anno−1 | corretto (la conclusione "credibilità per perilo inutile" regge) |
| 9 | abbinamento sinistri–righe con duplicati | filtro sulla data dentro il periodo; claim_id unici dopo l'abbinamento (0 duplicati su 160.432) | corretto |
| 35–36 | anno della riga ≠ anno di accadimento | AY = anno della riga per tutti i 160.432 sinistri | corretto |
| 4 | indirizzi duplicati | address_id unico | corretto |
| 50 | storia della polizza con i sinistri dell'anno | pol_prev_n = sinistri degli AY precedenti; le uniche 4.003 differenze sono polizze con 2 righe nello stesso anno, dove la storia esclude (prudente) anche la riga precedente dello stesso anno; mai incluso il sinistro della riga | nessun leakage |
| 16–19, 34, 67–71, 88–90 | sviluppo, trend, inflazione, recuperi e caricamento LL stimati su tutti gli anni anche nei fold | **vero**: è una scelta nota (le tabelle servono al modello finale). I fold sono un po' ottimisti sul livello, ma tutti i modelli confrontati usano le stesse tabelle | limite da scrivere nel report |

Altri controlli fatti senza Qwen: `prior_claims_5y` non cambia mai da un anno all'altro per la stessa polizza (dato dichiarato, nessun leakage); il GBM di Qwen usa 49–53 variabili ammesse, nessun identificativo né anno.
**Errore trovato nel GBM di Qwen:** vincolo monotono crescente su `vehicle_age`, mentre nei dati il costo relativo sale fino ai 5–8 anni (1,09) e poi scende (0,97 oltre i 12). Corretto in `qwen/experiment_v73_fix.py`, valutazione in coda.
