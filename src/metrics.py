"""Metrica della gara: Tweedie D² con power 1,5 rispetto a una previsione costante.

D² = 1 - D(y, mu) / D(y, mu0), con D la deviance Tweedie e mu0 la costante ottima
(media pesata). Coincide con sklearn.metrics.d2_tweedie_score(power=1.5).
"""
import numpy as np


def tweedie_deviance(y, mu, p=1.5, w=None):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    dev = 2 * (np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
               - y * np.power(mu, 1 - p) / (1 - p)
               + np.power(mu, 2 - p) / (2 - p))
    return float(np.average(dev, weights=w))


def d2_tweedie(y, mu, p=1.5, w=None):
    mu0 = np.full_like(np.asarray(y, float), np.average(y, weights=w))
    return 1 - tweedie_deviance(y, mu, p, w) / tweedie_deviance(y, mu0, p, w)


if __name__ == "__main__":
    from sklearn.metrics import d2_tweedie_score
    rng = np.random.default_rng(0)
    y = rng.gamma(0.1, 5000, 10_000) * (rng.random(10_000) < 0.1)
    mu = np.clip(y * 0.3 + 50, 1, None)
    print(d2_tweedie(y, mu), d2_tweedie_score(y, mu, power=1.5))
