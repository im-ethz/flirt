import multiprocessing
from datetime import timedelta

import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from .common import get_stats


def get_stat_features(data: pd.DataFrame, epoch_width: int = 60, num_cores: int = 0):
    """
    Computes several statistical and entropy-based time series features for each column in the provided DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        input time series
    epoch_width : int
        the epoch width (aka window size) in seconds to consider
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available

    Returns
    -------
    TS Features: pd.DataFrame
        A DataFrame containing all ststistical and entropy-based features.

    Notes
    -----
    DataFrame contains the following ACC features

        - **Statistical Features**: ts_entropy, ts_perm_entropy, ts_svd_entropy, ts_mean, \
        ts_min, ts_max, ts_ptp, ts_sum, ts_energy, ts_skewness, ts_kurtosis, ts_peaks, ts_rms, ts_lineintegral, \
        ts_n_above_mean, ts_n_below_mean, ts_iqr, ts_iqr_5_95, ts_pct_5, ts_pct_95

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> acc = flirt.reader.empatica.read_acc_file_into_df("ACC.csv")
    >>> acc_features = flirt.get_stat_features(acc, 60, 1)
    """

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    inputs = trange(0, len(data) - 1, 32, desc="Stat features")  # 32 Hz, calculate only once per second

    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(
            delayed(__ts_features)(data, epoch_width=epoch_width, i=k) for k in inputs)

    results = pd.DataFrame(list(filter(None, results)))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def __ts_features(data: pd.DataFrame, epoch_width: int, i: int):
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= epoch_width:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=epoch_width)
        results = {
            'datetime': max_timestamp,
        }

        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

        for column in relevant_data.columns:
            column_results = get_stats(relevant_data[column], column)
            results.update(column_results)

        return results

    else:
        return None
