import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from ..stats.common import get_stats


def get_temp_features(data: pd.Series, window_length: int = 60, window_step_size: float = 1.0, data_frequency: int = 4,
                     num_cores: int = 2):
    """
    Computes statistical Temperature features.

    Parameters
    ----------
    data : pd.DataFrame
        input temperature time series 
    window_length : int
        the window size in seconds to consider
    window_step_size : int
        the time step to shift each window
    data_frequency : int
        the frequency of the input signal
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available

    Returns
    -------
    Temperature Features: pd.DataFrame
        A DataFrame containing all ststistical features.

    Notes
    -----
    DataFrame contains the following Temperature features

        - **Statistical Features**: acc_entropy, acc_perm_entropy, acc_svd_entropy, acc_mean, \
        acc_min, acc_max, acc_ptp, acc_sum, acc_energy, acc_skewness, acc_kurtosis, acc_peaks, acc_rms, acc_lineintegral, \
        acc_n_above_mean, acc_n_below_mean, acc_iqr, acc_iqr_5_95, acc_pct_5, acc_pct_95

    Examples
    --------
    >>> temp_features = flirt.temp.get_temp_features(temp, 60)
    """

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    inputs = trange(0, len(data) - 1,
                    int(window_step_size * data_frequency))  # advance by window_step_size * data_frequency

    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(
            delayed(__get_stats)(data, window_length=window_length, i=k) for k in inputs)

    results = pd.DataFrame(list(filter(None, results)))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def __get_stats(data: pd.Series, window_length: int, i: int):
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= window_length:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)
        results = {
            'datetime': max_timestamp,
        }

        try:
            relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

            results.update(get_stats(np.ravel(relevant_data)))
        
        except Exception as e:
            pass

        return results

    else:
        return None
