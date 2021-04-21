import multiprocessing
from datetime import timedelta

import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange
from ..util import processing

from .common import get_stats


def get_stat_features(data: pd.DataFrame, window_length: int = 60, window_step_size: int = 1, data_frequency: int = 32,
                      entropies: bool = True, num_cores: int = 0):
    """
    Computes several statistical and entropy-based time series features for each column in the provided DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        input time series
    window_length : int
        the epoch width (aka window size) in seconds to consider
    entropies : bool
        whether to calculate entropy features
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available

    Returns
    -------
    TS Features: pd.DataFrame
        A DataFrame containing all ststistical and entropy-based features.

    Notes
    -----
    DataFrame contains the following stat features

        - **Statistical Features**: entropy (optional), perm_entropy (optional), svd_entropy (optional), mean, \
        min, max, ptp, sum, energy, skewness, kurtosis, peaks, rms, lineintegral, \
        n_above_mean, n_below_mean, iqr, iqr_5_95, pct_5, pct_95

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> acc = flirt.reader.empatica.read_acc_file_into_df("ACC.csv")
    >>> acc_features = flirt.get_stat_features(acc, 60, 1, entropies=False)
    """

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    input_data = data.copy()
    # advance by window_step_size * data_frequency
    inputs = trange(0, len(input_data) - 1, window_step_size * data_frequency, desc="Stat features")

    def process(memmap_data):
        with Parallel(n_jobs=num_cores, max_nbytes=None) as parallel:
            return parallel(
                delayed(__ts_features)(memmap_data, epoch_width=window_length, i=k, entropies=entropies)
                for k in inputs)
    results = processing.memmap_auto(input_data, process)

    results = pd.DataFrame(list(filter(None, results)))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def __ts_features(data: pd.DataFrame, epoch_width: int, i: int, entropies: bool = True):
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= epoch_width:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=epoch_width)
        results = {
            'datetime': max_timestamp,
        }

        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

        for column in relevant_data.columns:
            column_results = get_stats(relevant_data[column], column, entropies=entropies)
            results.update(column_results)

        return results

    else:
        return None
