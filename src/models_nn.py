"""Two additional models (17/10): elastic-net GLM (glmnet style) and CANN.

glm_enet  — Tweedie GLM (p = 1.5, log link) with elastic-net penalty (L1 + L2) via glum, like glmnet in R.
            Numeric variables in 10 bins (step curve, like a tariff), categoricals one-hot encoded,
            make_model left raw: the penalty decides how much to trust each car model.
            Soil × drought interaction. Fit: response = cost / exposure, weight = exposure.
cann      — Combined Actuarial Neural Network (Wüthrich & Merz): the freq×sev prediction (fs2_all_ll) is the base,
            a small network learns only the multiplicative correction exp(g(x)); g starts at 0, so the network
            initially IS the actuarial model and moves only where the data require it. Tweedie loss p = 1.5,
            early stopping on 10% of the training policies.
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


def _design_old(train, valid, line, nbins=10, min_count=2000):
    raise NotImplementedError   # Python-string version: 6 GB on motor data, replaced on 18/10


def _cat_pair(fa, fb, min_count):
    """fa, fb: categorical Series (same labels). Merges rare levels, aligns validation."""
    vc = fa.value_counts()
    rare = list(vc[vc < min_count].index)
    if rare:   # levels with few rows: a single large loss would be enough to produce a huge coefficient
        if "rare" not in fa.cat.categories:
            fa = fa.cat.add_categories("rare")
        fa = fa.where(~fa.isin(rare), "rare")
        fb = fb.astype(object).where(~fb.astype(object).isin(rare), "rare")
    fa = fa.cat.remove_unused_categories()
    cats = list(fa.cat.categories)
    fb = pd.Categorical(pd.Series(fb, dtype=object).where(pd.Series(fb, dtype=object).isin(cats), cats[0]), categories=cats)
    return pd.Categorical(fa, categories=cats), fb


def _design(train, valid, line, nbins=10, min_count=2000):
    """Same design as before (quantile bins, rare levels merged), built with categories instead of strings."""
    cols = [c for c in M._features(train, line)]
    Xtr, Xva = {}, {}
    for c in cols:
        a, b = train[c], valid[c]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a) and a.nunique() > nbins:
            edges = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, nbins + 1)[1:-1]))
            fa = np.where(a.isna(), -1, np.searchsorted(edges, a.astype(float).fillna(0), side="right"))
            fb = np.where(b.isna(), -1, np.searchsorted(edges, b.astype(float).fillna(0), side="right"))
            fa, fb = pd.Series(fa, dtype="category"), pd.Series(fb)
        else:
            fa = a.astype("string").fillna("na").astype("category").reset_index(drop=True)
            fb = b.astype("string").fillna("na").reset_index(drop=True)
        Xtr[c], Xva[c] = _cat_pair(fa, fb, min_count)
    if "soil_type" in train.columns and "dry_summer_index" in train.columns:
        q = np.nanquantile(train["dry_summer_index"], [0.25, 0.5, 0.75])
        fa = train["soil_type"].astype("string").fillna("na").str.cat(pd.Series(np.searchsorted(q, train["dry_summer_index"].fillna(0)), index=train.index).astype("string"), sep="|")
        fb = valid["soil_type"].astype("string").fillna("na").str.cat(pd.Series(np.searchsorted(q, valid["dry_summer_index"].fillna(0)), index=valid.index).astype("string"), sep="|")
        Xtr["soil_x_dry"], Xva["soil_x_dry"] = _cat_pair(fa.astype("category").reset_index(drop=True), fb.reset_index(drop=True), min_count)
    return pd.DataFrame(Xtr), pd.DataFrame(Xva)


def glm_enet(train, valid, line, alpha=None, l1_ratio=ENET_L1):
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    Xtr, Xva = _design(train, valid, line)
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    y = train[P.TARGET].to_numpy()
    if alpha is None:   # penalty selection on 10% of the training policies
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
    # robust transform: rank → approximate normal scale on training quantiles
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = np.clip((A - mu) / sd, -5, 5), np.clip((B - mu) / sd, -5, 5)
    C_tr, C_va, card = [], [], []
    for c in cat:
        a = train[c].astype(object).where(train[c].notna(), "na").astype(str)
        b = valid[c].astype(object).where(valid[c].notna(), "na").astype(str)
        vc = a.value_counts(); keep = vc[vc >= 50].index
        lev = {v: i + 1 for i, v in enumerate(keep)}  # 0 = rare / new
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
            torch.nn.init.zeros_(self.out.weight); torch.nn.init.zeros_(self.out.bias)   # starts from the actuarial model

        def forward(self, xn, xc):
            z = [xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
            return self.out(self.body(torch.cat(z, 1))).squeeze(1)

    def tweedie_loss(mu, y, p=1.5):
        return (-y * mu.pow(1 - p) / (1 - p) + mu.pow(2 - p) / (2 - p)).mean()

    # internal validation: 10% of the training policies
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
