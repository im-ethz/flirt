import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, iqr

from .entropy import get_entropies

FUNCTIONS = {
    'mean': np.mean,
    'std': np.std,
    'min': np.min,
    'max': np.max,
    'ptp': np.ptp,
    'sum': np.sum,
    'energy': lambda x: np.sum(x ** 2),
    'skewness': skew,
    'kurtosis': kurtosis,
    'peaks': lambda x: len(find_peaks(x, prominence=0.9)[0]),
    'rms': lambda x: np.sqrt(np.sum(x ** 2) / len(x)),
    'lineintegral': lambda x: np.abs(np.diff(x)).sum(),
    'n_above_mean': lambda x: np.sum(x > np.mean(x)),
    'n_below_mean': lambda x: np.sum(x < np.mean(x)),
    'n_sign_changes': lambda x: np.sum(np.diff(np.sign(x)) != 0),
    'iqr': iqr,
    'iqr_5_95': lambda x: iqr(x, rng=(5, 95)),
    'pct_5': lambda x: np.percentile(x, 5),
    'pct_95': lambda x: np.percentile(x, 95),
}


def get_stats(data, key_suffix: str = None, entropies: bool = True):
    data = np.asarray(data)
    results = {}
    if len(data) > 0:
        for key, value in FUNCTIONS.items():
            results[key] = value(data)
    else:
        for key in FUNCTIONS.keys():
            results[key] = np.nan

    # Update with entropies
    if entropies:
        results.update(get_entropies(data))

    if key_suffix is not None:
        results = {key_suffix+'_'+k: v for k, v in results.items()}
    return results
