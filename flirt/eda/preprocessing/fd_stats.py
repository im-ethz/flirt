import numpy as np
import scipy
from scipy.signal import welch
from scipy.fft import fft
from scipy.stats import kurtosis, iqr

def bandpower(f, Pxx, fmin, fmax):
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

FUNCTIONS = {
    'fd_sma': lambda x: np.sum(np.abs(x-np.mean(x)))/len(x),
    'fd_energy': lambda x: np.mean(np.abs(x) ** 2),
    'fd_spectral_power': lambda x: welch(x, fs=4, nperseg=len(x), scaling='spectrum'),
    'fd_kurtosis': lambda x: kurtosis(np.abs(x)),
    'fd_iqr': lambda x: iqr(np.abs(x)),
}


def get_fd_stats(data, key_suffix: str = None):
    data = np.asarray(data)
    results = {}
    if len(data) > 0:
        for key, value in FUNCTIONS.items():
            if key=='fd_spectral_power':
                freq, power = value(data)
                results['fd_varPower'] = np.sum(power)*(freq[1]-freq[0])
                for i in [0.05, 0.15, 0.25, 0.35, 0.45]:
                    results['fd_Power_'+str(i)] = bandpower(freq, power, i, i+0.1)
            else:
                data = fft(data)
                results[key] = value(data)
    else:
        for key in FUNCTIONS.keys():
            results[key] = np.nan

    if key_suffix is not None:
        results = {key_suffix+'_'+k: v for k, v in results.items()}
    return results
