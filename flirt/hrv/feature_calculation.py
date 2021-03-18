import datetime
import multiprocessing
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .features.data_utils import DomainFeatures
from .features.fd_features import FdFeatures
from .features.nl_features import NonLinearFeatures
from .features.td_features import TdFeatures
from ..stats.common import get_stats
from ..util import processing

# disable astropy warnings
try:
    from astropy.utils.exceptions import AstropyWarning
    import warnings

    warnings.simplefilter('ignore', category=AstropyWarning)
except:
    pass

# disable astropy warnings
try:
    from astropy.utils.exceptions import AstropyWarning
    import warnings

    warnings.simplefilter('ignore', category=AstropyWarning)
except:
    pass


class StatFeatures(DomainFeatures):
    def __get_type__(self) -> str:
        return "Statistical"

    def __generate__(self, data: np.array) -> dict:
        return get_stats(data, 'hrv')


FEATURE_FUNCTIONS = {
    'td': TdFeatures(),
    'fd': FdFeatures(sampling_frequency=1, method='lomb'),
    'stat': StatFeatures(),
    'nl': NonLinearFeatures()
}


def get_hrv_features(data: pd.Series, window_length: int = 180, window_step_size: int = 1,
                     domains: List[str] = ['td', 'fd', 'stat'], threshold: float = 0.2,
                     clean_data: bool = True, num_cores: int = 0):
    """
    Computes HRV features for different domains (time-domain, frequency-domain, non-linear, statistical).

    Parameters
    ----------
    data : pd.Series
        input IBIs in milliseconds
    window_length : int
        the epoch width (aka window size) in seconds to consider
    window_step_size : int
        the step size for the sliding window in seconds
    domains : List[str]
        domains to calculate (possible values: "td", "fd", "nl", "stat")
    threshold : float
        the relative portion of IBIs that need to be present in a window for processing. \
        Otherwise, the window is dropped
    clean_data : bool, optional
        whether obviously invalid IBIs should be removed before processing, by default True
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available

    Returns
    -------
    HRV Features: pd.DataFrame
        A DataFrame containing all chosen features (i.e., time domain, frequency domain, non-linear features, \
        and/or statistical features)

    Notes
    -----
    DataFrame contains the following HRV features

        - **Time Domain**: rmssd, sdnn, sdsd, nni_50, pnni_50, nni_20, pnni_20, \
        mean_hr, max_hr, min_hr, std_hr, cvsd, cvnni

        - **Frequency Domain**: tital_power, hf, vlf, lf, lf_hf_ratio, lfnu, hfnu

        - **Non-Linear Features**: SD1, SD2, SD2SD1, CSI, CVI, CSI_Modified

        - **Statistical Features**: ibi_entropy, ibi_perm_entropy, ibi_svd_entropy, ibi_ibi_mean, \
        ibi_min, ibi_max, ibi_ptp, ibi_sum, ibi_energy, ibi_skewness, ibi_kurtosis, ibi_peaks, ibi_rms, \
        ibi_lineintegral, ibi_n_above_mean, ibi_n_below_mean, ibi_iqr, ibi_iqr_5_95, ibi_pct_5, ibi_pct_95

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> ibis = flirt.reader.empatica.read_ibi_file_into_df("IBI.csv")
    >>> hrv_features = flirt.get_hrv_features(ibis['ibi'], 60, 1, ["td", "fd", "stat"], 0.8)
    """

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    if clean_data:
        # print("Cleaning data...")
        clean_data = __clean_artifacts(data.copy())
    else:
        # print("Not cleaning data")
        clean_data = data.copy()

    # ensure we have a DatetimeIndex, needed for interpolation
    if not isinstance(clean_data.index, pd.DatetimeIndex):
        clean_data.index = pd.DatetimeIndex(clean_data.index)

    clean_data = clean_data[~clean_data.index.duplicated()]

    window_length_timedelta = pd.to_timedelta(window_length, unit='s')
    window_step_size_timedelta = pd.to_timedelta(window_step_size, unit='s')

    target_index = pd.date_range(start=clean_data.index[0].floor('s'),
                                 end=clean_data.index[-1].ceil('s') - window_length_timedelta,
                                 freq=window_step_size_timedelta)

    def process(memmap_data) -> pd.DataFrame:
        with Parallel(n_jobs=num_cores, max_nbytes=None) as parallel:
            for domain in domains:
                if domain not in FEATURE_FUNCTIONS.keys():
                    raise ValueError("invalid feature domain: " + domain)

            return __generate_features_for_domain(memmap_data, target_index, window_length_timedelta, threshold,
                                                  feature_functions=[FEATURE_FUNCTIONS.get(x) for x in domains],
                                                  parallel=parallel)

    features = processing.memmap_auto(clean_data, process)

    # only interpolate if there are overlapping time windows
    if window_length_timedelta > window_step_size_timedelta:
        limit = max(1, int((1 - threshold) * (window_length / window_step_size)))
        features.interpolate(method='time', limit=limit, inplace=True)
    return features


def __clean_artifacts(data: pd.Series, threshold=0.2) -> pd.Series:
    """
    Cleans obviously illegal IBI values (artefacts) from a list

    Parameters
    ----------
    data : pd.Series
        the IBI list
    threshold : float, optional
        the maximum relative deviation between subsequent intervals, by default 0.2

    Returns
    -------
    pd.Series
        the cleaned IBIs
    """

    # Artifact detection - Statistical
    # for index in trange(data.shape[0]):
    #    # Remove RR intervals that differ more than 20% from the previous one
    #    if np.abs(data.iloc[index] - data.iloc[index - 1]) > 0.2 * data.iloc[index]:
    #        data.iloc[index] = np.nan

    # efficiency instead of loop ;-)
    diff = data.diff().abs()
    drop_indices = diff > threshold * data
    if drop_indices.any():
        data.drop(data[drop_indices].index, inplace=True)
    drop_indices = (data < 250) | (data > 2000)
    if drop_indices.any():
        data.drop(data[drop_indices].index, inplace=True)  # drop by bpm > 240 or bpm < 30
    data.dropna(inplace=True)  # just to be sure

    return data


def __generate_features_for_domain(clean_data: pd.Series, target_index: pd.DatetimeIndex,
                                   window_length: datetime.timedelta, threshold: float,
                                   feature_functions: List[DomainFeatures], parallel: Parallel) -> pd.DataFrame:
    inputs = tqdm(target_index, desc="HRV features")

    features = parallel(delayed(__calculate_hrv_features)
                        (clean_data, start_datetime=k, window_length=window_length,
                         threshold=threshold,
                         feature_functions=feature_functions)
                        for k in inputs)
    features = pd.DataFrame(list(filter(None, features)))
    if not features.empty:
        features.set_index('datetime', inplace=True)
        features.sort_index(inplace=True)
    return features


def __calculate_hrv_features(data: pd.Series, window_length: datetime.timedelta, start_datetime: datetime.datetime,
                             threshold: float, feature_functions: List[DomainFeatures]):
    relevant_data = data.loc[(data.index >= start_datetime) & (data.index < start_datetime + window_length)]

    return_val = {'datetime': start_datetime + window_length, "num_ibis": len(relevant_data)}
    # first check if there is at least one IBI in the epoch
    if len(relevant_data) > 0:
        expected_length = (window_length.total_seconds() / (relevant_data.mean() / 1000))
        actual_length = len(relevant_data)

        if actual_length >= (expected_length * threshold):
            for feature_function in feature_functions:
                return_val.update(feature_function.__generate__(relevant_data))

    return return_val
