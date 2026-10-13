"""Small statistics helpers for live code (kept free of the experiment modules)."""
import numpy as np
import pandas as pd

from premodel import config


def date_block_bootstrap(values: np.ndarray, dates, rng: np.random.Generator | None = None,
                         n: int = config.BOOTSTRAP_SAMPLES) -> dict:
    """Mean of per-match `values` with a 95% CI, resampling whole match dates (matches on
    one date share a model fit and are not independent). Same method as the experiments."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n_matches": 0,
                "n_dates": 0}
    rng = rng if rng is not None else np.random.default_rng(config.SEED)
    codes, uniq = pd.factorize(pd.Series(pd.DatetimeIndex(dates)))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_matches": int(len(values)), "n_dates": int(len(uniq))}
