import numpy as np
from scipy.stats import entropy

from ..lib.entropy import perm_entropy, svd_entropy

ENTROPIES = {
    'entropy': lambda x, order, delay: entropy(x),
    'perm_entropy': lambda x, order, delay: perm_entropy(x, order=order, delay=delay),
    # 'spectral_entropy': lambda x, order, delay: spectral_entropy(x),
    'svd_entropy': lambda x, order, delay: svd_entropy(x, order=order, delay=delay),

    # very compute intensive!
    # 'app_entropy': lambda x, order, delay: app_entropy(x, order=order),
    # 'sample_entropy': lambda x, order, delay: sample_entropy(x, order=order),
}


def get_entropies(data, emb_dim: int = 2, tau: int = 3):
    results = {}
    if len(data) > emb_dim * tau:
        for key, value in ENTROPIES.items():
            results[key] = value(data, emb_dim, tau)
    else:
        for key in ENTROPIES.keys():
            results[key] = np.nan

    return results
