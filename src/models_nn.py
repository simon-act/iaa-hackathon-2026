"""Due modelli in più richiesti da Simone (17/10): GLM elastic net (stile glmnet) e CANN.

glm_enet  — GLM Tweedie (p = 1,5, link log) con penalità elastic net (L1 + L2) via glum, come glmnet in R.
            Variabili numeriche in 10 classi (curva a gradini, come una tariffa), categoriche in one-hot,
            make_model lasciata grezza: è la penalità a decidere quanto credere a ogni modello d'auto.
            Interazione suolo × siccità. Taratura: risposta = costo / esposizione, peso = esposizione.
cann      — Combined Actuarial Neural Network (Wüthrich & Merz): la previsione del freq×sev (fs2_all_ll) è la base,
            una piccola rete impara solo la correzione moltiplicativa exp(g(x)); g parte da 0, quindi la rete
            all'inizio È il modello attuariale e si sposta solo dove i dati lo chiedono. Perdita Tweedie p = 1,5,
            arresto anticipato su un 10% di polizze del training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402

ENET_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
ENET_L1 = 0.5


def _design(train, valid, line, nbins=10, min_count=2000):
    cols = [c for c in M._features(train, line)]
    Xtr, Xva = {}, {}
    for c in cols:
        a, b = train[c], valid[c]
        if pd.api.types.is_numeric_dtype(a) and a.nunique() > nbins:
            edges = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, nbins + 1)[1:-1]))
            fa = np.where(a.isna(), "na", np.searchsorted(edges, a.astype(float).fillna(0), side="right").astype(str))
            fb = np.where(b.isna(), "na", np.searchsorted(edges, b.astype(float).fillna(0), side="right").astype(str))
        else:
            fa = a.astype(object).where(a.notna(), "na").astype(str).to_numpy()
            fb = b.astype(object).where(b.notna(), "na").astype(str).to_numpy()
        Xtr[c], Xva[c] = fa, fb
    if "soil_type" in train.columns and "dry_summer_index" in train.columns:
        q = np.nanquantile(train["dry_summer_index"], [0.25, 0.5, 0.75])
        Xtr["soil_x_dry"] = [f"{a}|{b}" for a, b in zip(train["soil_type"].astype(str), np.searchsorted(q, train["dry_summer_index"].fillna(0)))]
        Xva["soil_x_dry"] = [f"{a}|{b}" for a, b in zip(valid["soil_type"].astype(str), np.searchsorted(q, valid["dry_summer_index"].fillna(0)))]
    Xtr, Xva = pd.DataFrame(Xtr), pd.DataFrame(Xva)
    for c in Xtr.columns:
        vc = Xtr[c].value_counts()
        rare = set(vc[vc < min_count].index)
        if rare:   # livelli con poche righe: un grande sinistro basterebbe a dare un coefficiente enorme
            Xtr[c] = Xtr[c].where(~Xtr[c].isin(rare), "rare"); Xva[c] = Xva[c].where(~Xva[c].isin(rare), "rare")
        cats = sorted(set(Xtr[c]))
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        Xva[c] = pd.Categorical(Xva[c].where(Xva[c].isin(cats), cats[0]), categories=cats)
    return Xtr, Xva


def glm_enet(train, valid, line, alpha=None, l1_ratio=ENET_L1):
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    Xtr, Xva = _design(train, valid, line)
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    y = train[P.TARGET].to_numpy()
    if alpha is None:   # scelta della penalità su un 10% di polizze del training
        h = pd.util.hash_array(train["policy_id"].to_numpy()) % 10 == 3
        best = (-1e9, None)
        for a in ENET_GRID:
            g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=a, l1_ratio=l1_ratio,
                                           max_iter=300, drop_first=False).fit(Xtr[~h], (y / e_tr)[~h], sample_weight=e_tr[~h])
            pv = g.predict(Xtr[h]) * e_tr[h]
            sc = P.d2(y[h], pv * y[~h].sum() / (g.predict(Xtr[~h]) * e_tr[~h]).sum())
            if sc > best[0]:
                best = (sc, a)
        alpha = best[1]
    glm_enet.last_alpha = alpha
    m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha,
                                   l1_ratio=l1_ratio, max_iter=300, drop_first=False)
    m.fit(Xtr, y / e_tr, sample_weight=e_tr)
    p_tr, p_va = m.predict(Xtr) * e_tr, m.predict(Xva) * e_va
    glm_enet.last_nonzero = int((np.abs(m.coef_) > 1e-8).sum()), int(m.coef_.size)
    return p_va * (y.sum() / p_tr.sum())


# ---------------------------------------------------------------- CANN
def _nn_inputs(train, valid, line):
    cols = M._features(train, line)
    num, cat = [], []
    for c in cols:
        (num if pd.api.types.is_numeric_dtype(train[c]) and train[c].dtype != bool and train[c].nunique() > 2 else cat).append(c)
    def numblock(df):
        out = []
        for c in num:
            x = df[c].astype(float)
            out.append(x.to_numpy()); out.append(x.isna().to_numpy(float))
        return np.column_stack(out) if out else np.zeros((len(df), 0))
    A, B = numblock(train), numblock(valid)
    med = np.nanmedian(A, axis=0); A = np.where(np.isnan(A), med, A); B = np.where(np.isnan(B), med, B)
    # trasformazione robusta: rango → scala normale approssimata sui quantili del training
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = np.clip((A - mu) / sd, -5, 5), np.clip((B - mu) / sd, -5, 5)
    C_tr, C_va, card = [], [], []
    for c in cat:
        a = train[c].astype(object).where(train[c].notna(), "na").astype(str)
        b = valid[c].astype(object).where(valid[c].notna(), "na").astype(str)
        vc = a.value_counts(); keep = vc[vc >= 50].index
        lev = {v: i + 1 for i, v in enumerate(keep)}  # 0 = raro / nuovo
        C_tr.append(a.map(lev).fillna(0).astype(int).to_numpy()); C_va.append(b.map(lev).fillna(0).astype(int).to_numpy())
        card.append(len(lev) + 1)
    C_tr = np.column_stack(C_tr) if C_tr else np.zeros((len(train), 0), int)
    C_va = np.column_stack(C_va) if C_va else np.zeros((len(valid), 0), int)
    return A.astype(np.float32), B.astype(np.float32), C_tr, C_va, card


def cann(train, valid, line, epochs=12, hidden=(32, 16), lr=2e-3, wd=1e-4, seed=0):
    import torch
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed)
    both = pd.concat([train, valid], ignore_index=True)
    base_all = np.asarray(M._fs2(train, both, line, ll_si=True), float)
    b_tr, b_va = base_all[:len(train)], base_all[len(train):]
    A, B, Ctr, Cva, card = _nn_inputs(train, valid, line)
    y = train[P.TARGET].to_numpy().astype(np.float32)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.ModuleList([torch.nn.Embedding(k, min(8, max(2, k // 2))) for k in card])
            d = A.shape[1] + sum(min(8, max(2, k // 2)) for k in card)
            layers, prev = [], d
            for h in hidden:
                layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]; prev = h
            self.body = torch.nn.Sequential(*layers)
            self.out = torch.nn.Linear(prev, 1)
            torch.nn.init.zeros_(self.out.weight); torch.nn.init.zeros_(self.out.bias)   # parte dal modello attuariale

        def forward(self, xn, xc):
            z = [xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
            return self.out(self.body(torch.cat(z, 1))).squeeze(1)

    def tweedie_loss(mu, y, p=1.5):
        return (-y * mu.pow(1 - p) / (1 - p) + mu.pow(2 - p) / (2 - p)).mean()

    # validazione interna: 10% delle polizze del training
    pol = train["policy_id"].to_numpy()
    h = pd.util.hash_array(pol) % 10 == 3
    t_idx, v_idx = np.where(~h)[0], np.where(h)[0]
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    An, Cn, Yn, Bn = T(A), T(Ctr.astype(np.int64)), T(y), T(np.log(b_tr).astype(np.float32))
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e18, None, 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(t_idx)
        for i in range(0, len(perm), 8192):
            j = T(perm[i:i + 8192])
            mu = torch.exp(Bn[j] + net(An[j], Cn[j]))
            loss = tweedie_loss(mu, Yn[j]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            jv = T(v_idx); vl = float(tweedie_loss(torch.exp(Bn[jv] + net(An[jv], Cn[jv])), Yn[jv]))
        if vl < best - 1e-7:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        g_tr = net(An, Cn).numpy(); g_va = net(T(B), T(Cva.astype(np.int64))).numpy()
    p_tr, p_va = b_tr * np.exp(g_tr), b_va * np.exp(g_va)
    cann.last_info = dict(epochs=ep + 1, g_sd_valid=float(np.std(g_va)))
    return p_va * (y.sum() / p_tr.sum())
