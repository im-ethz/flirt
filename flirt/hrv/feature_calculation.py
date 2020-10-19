import multiprocessing
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from .features.data_utils import DomainFeatures
from .features.fd_features import FdFeatures
from .features.nl_features import NonLinearFeatures
from .features.td_features import TdFeatures
from ..stats.common import get_stats

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


feature_functions = {
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

    calculated_features = {}

    with Parallel(n_jobs=num_cores) as parallel:
        for domain in domains:
            if domain not in feature_functions.keys():
                raise ValueError("invalid feature domain: " + domain)

            #print("Calculate %s features" % domain)

            feat = __generate_features_for_domain(clean_data, window_length, threshold,
                                                  feature_function=feature_functions[domain], parallel=parallel)
            calculated_features[domain] = feat

    features = pd.concat(calculated_features.values(), axis=1, sort=True)

    target_index = pd.date_range(start=features.iloc[0].name.ceil('s'),
                                 end=features.iloc[-1].name.floor('s'), freq='%ds' % (window_step_size), tz='UTC')

    features = features.reindex(index=features.index.union(target_index))

    features.interpolate(method='time', limit=int((1 - threshold) * (window_length / window_step_size)), inplace=True)
    features = features.reindex(target_index)
    features.index.name = 'datetime'

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
    data.drop(data[diff > threshold * data].index, inplace=True)
    data.drop(data[(data < 250) | (data > 2000)].index, inplace=True)  # drop by bpm > 240 or bpm < 30
    data.dropna(inplace=True)  # just to be sure

    return data


def __generate_features_for_domain(clean_data: pd.Series, window_length: int, threshold: float,
                                   feature_function: DomainFeatures, parallel: Parallel) -> pd.DataFrame:
    inputs = trange(len(clean_data) - 1, desc="HRV %s features " % feature_function.__get_type__())

    features = parallel(delayed(__calculate_hrv_features)
                        (clean_data, i=k, window_length=window_length,
                         threshold=threshold,
                         feature_function=feature_function)
                        for k in inputs)
    features = pd.DataFrame(list(filter(None, features)))
    if not features.empty:
        features.set_index('datetime', inplace=True)
        features.dropna(inplace=True)
        features.sort_index(inplace=True)
    return features


def __calculate_hrv_features(data: pd.Series, window_length: int, i: int, threshold: float,
                             feature_function: DomainFeatures):
    # first check if there is at least one IBI in the epoch
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= window_length:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)
        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

        expected_length = (window_length / (relevant_data.mean() / 1000))
        actual_length = len(relevant_data)

        if actual_length >= (expected_length * threshold):
            return {'datetime': max_timestamp, **feature_function.__generate__(relevant_data)}

    return None
