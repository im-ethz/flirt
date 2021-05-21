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
    """ 
    Function that computes the Mel-Frequency Cepstrum Components (MFCC) features.
    
    Parameters
    -----------
    data: np.ndarray or pd.Series
        ACC data onto which to compute the features, can be the raw data, the filtered data, the phasic or the tonic component. 
    key_suffix: str, optional
        Suffix to place in-front of feature name.
        
    Returns
    --------
    dict
        Returns a dictionary with the following MFCC features: mean, standard deviation, median, skewnes, kurtosis, inter-quartile range.
    """
    
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