import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange
from ..util import processing

from .preprocessing import data_utils, LowPassFilter, ParticleFilter
from ..stats.common import get_stats
from .preprocessing import get_mfcc_stats, get_fd_stats


def get_acc_features(data: pd.DataFrame, window_length: int = 60, window_step_size: float = 1,
                     data_frequency: int = 32, num_cores: int = 0,
                     preprocessor: data_utils.Preprocessor = LowPassFilter()):
    """
    Computes statistical ACC features based on the l2-norm of the x-, y-, and z- acceleration.

    Parameters
    ----------
    data : pd.DataFrame
        input ACC time series in x-, y-, and z- direction
    window_length : int
        the window size in seconds to consider
    window_step_size : int
        the time step to shift each window
    data_frequency : int
        the frequency of the input signal
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available
    preprocessor: class, optional
        the method chosen to clean the data: low-pass filtering, particle filtering

    Returns
    -------
    ACC Features: pd.DataFrame
        A DataFrame containing time-domain, peak and frequency-domain aggregation features.

    Notes
    -----
    DataFrame contains the following ACC features

        - **Statistical Features**: acc_entropy, acc_perm_entropy, acc_svd_entropy, acc_mean, \
        acc_min, acc_max, acc_ptp, acc_sum, acc_energy, acc_skewness, acc_kurtosis, acc_peaks, acc_rms, acc_lineintegral, \
        acc_n_above_mean, acc_n_below_mean, acc_iqr, acc_iqr_5_95, acc_pct_5, acc_pct_95
        - **Frequency-Domain Features**: fd_sma, fd_energy, fd_varPower, fd_Power_0.05, fd_Power_0.15, fd_Power_0.25, fd_Power_0.35, \
        fd_Power_0.45, fd_kurtosis, fd_iqr, 
        - **Time-Frequency-Domain Features**: mfcc_mean, mfcc_std, mfcc_median, mfcc_skewness, mfcc_kurtosis, mfcc_iqr

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> acc = flirt.reader.empatica.read_acc_file_into_df("ACC.csv")
    >>> acc_features = flirt.get_acc_features(acc, 60)
    """

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    input_data = data.copy()

    # Filter Data in all three ACC directions: x, y, z
    for column in input_data.columns:
        input_data[column] = preprocessor.__process__(data[column])

    # Find ACC norm after filtering
    input_data['l2'] = np.linalg.norm(input_data.to_numpy(), axis=1)

    # ensure we have a DatetimeIndex, needed for calculation
    if not isinstance(input_data.index, pd.DatetimeIndex):
        input_data.index = pd.DatetimeIndex(input_data.index)

    inputs = trange(0, len(input_data) - 1,
                    window_step_size * data_frequency,
                    desc="ACC features")  # advance by window_step_size * data_frequency

    def process(memmap_data) -> dict:
        with Parallel(n_jobs=num_cores, max_nbytes=None) as parallel:
            return parallel(delayed(__get_l2_stats)(memmap_data, window_length=window_length, i=k) for k in inputs)

    results = processing.memmap_auto(input_data, process)

    results = pd.DataFrame(list(filter(None, results)))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def __get_l2_stats(data: pd.DataFrame, window_length: int, i: int):
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= window_length:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)
        results = {
            'datetime': max_timestamp,
        }

        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

        for column in relevant_data.columns:
            column_results_td_stats = get_stats(relevant_data[column], column)
            column_results_fd_stats = get_fd_stats(relevant_data[column], column)
            column_results_mfcc_stats = get_mfcc_stats(relevant_data[column], column)
            results.update(column_results_td_stats)
            results.update(column_results_fd_stats)
            results.update(column_results_mfcc_stats)

        return results

    else:
        return None
