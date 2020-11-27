import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import skew, kurtosis, iqr
    
FUNCTIONS = {
    'mfcc_mean': np.mean,
    'mfcc_std': np.std,
    'mfcc_median': np.median,
    'mfcc_skewness': skew,
    'mfcc_kurtosis': kurtosis,
    'mfcc_iqr': iqr,
}

def __real_cestrum(data):
    spectrum = fft(data)
    ceps = ifft(np.log(np.abs(spectrum))).real
    return ceps

def get_MFCC_stats(data, key_suffix: str = None):
    data = np.asarray(data)
    results = {}
    if len(data) > 0:
        data = __real_cestrum(data)
        for key, value in FUNCTIONS.items():
            results[key] = value(data)
    else:
        for key in FUNCTIONS.keys():
            results[key] = np.nan

    if key_suffix is not None:
        results = {key_suffix+'_'+k: v for k, v in results.items()}
    return results