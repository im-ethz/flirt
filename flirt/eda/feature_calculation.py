import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from .peak_features import __compute_peaks_features
from .preprocessing import data_utils, CvxEda, LowPassFilter
from ..stats.common import get_stats


def get_eda_features(data: pd.Series, data_frequency: int = 4, window_length: int = 60, window_step_size: int = 1,
                 offset: int = 1, start_WT: int = 3, end_WT: int = 10, thres: float = 0.01, num_cores=0,
                 preprocessor: data_utils.Preprocessor = LowPassFilter(),
                 signal_decomposition: data_utils.SignalDecomposition = CvxEda()):
    """
    This function computes several statistical and entropy-based features for the phasic and tonic components of the EDA signal.

    Parameters
    ----------
    data : pd.Series
        raw EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`
    window_length: int, optional
        length of window to slide over the data, to compute features (in seconds)
    window_step_size:
        step-size taken to move the sliding window (in seconds)
    data_frequency : int, optional
        the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
    offset: int, optional
        the number of rising samples and falling samples after a peak needed to be counted as a peak
    start_WT: int, optional
        maximum number of seconds before the apex of a peak that is the "start" of the peak
    end_WT: int, optional
        maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
    thres: float, optional
        the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available
    preprocessor: class, optional
        the method chosen to clean the data: low-pass filtering or Kalman filtering or artifact-detection with low-pass filtering

    Returns
    -------
    pd.DataFrame
        dataframe containing all statistical, peak and entropy-based features, index is a list of timestamps according to the window step size

    Notes
    -----
    The output dataframe contains the following EDA features computed for both the tonic and phasic components

        - **Statistical Features**: mean, std, min, max, ptp, sum, energy, skewness, kurtosis, \
        peaks, rms, lineintegral, n_above_mean, n_below_mean, n_sign_changes, iqr, iqr_5_95, pct_5, pct_95
        - **Peak Features**: peaks_p, rise_time_p, max_deriv_p, amp_p, decay_time_p, SCR_width_p, auc_p
        - **Entropy Features**: entropy, perm_entropy, svd_entropy

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> import flirt.eda
    >>> path = './empatica/1575615731_A12345'
    >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
    >>> eda_features = flirt.eda.get_features(eda['eda'], data_frequency=4, window_length=60, window_step_size=1)
    """

    # Get number of cores
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    # use only every fourth value
    inputs = trange(0, len(data) - 1, window_step_size * data_frequency)

    # Filter Data
    filtered_dataframe = preprocessor.__process__(data)

    # Decompose Data
    phasic_data, tonic_data = signal_decomposition.__process__(filtered_dataframe)

    # Get features
    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(delayed(__get_features_per_window)(phasic_data, tonic_data, window_length=window_length,
                                                              i=k, sampling_frequency=data_frequency, offset=offset,
                                                              start_WT=start_WT, end_WT=end_WT, thres=thres) for k in inputs)

    # results = pd.DataFrame(list(filter(None, results)))
    results = pd.DataFrame(list(results))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    print('- Feature generation completed')

    return results


def __get_features_per_window(phasic_data: pd.Series, tonic_data: pd.Series, window_length: int, i: int, sampling_frequency: int = 4,
                              offset: int = 1, start_WT: int = 3, end_WT: int = 10, thres: float = 0.01):
    

    if pd.Timedelta(phasic_data.index[i + 1] - phasic_data.index[i]).total_seconds() <= window_length:
        min_timestamp = phasic_data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)

        results = {
            'datetime': max_timestamp,
        }
        
        try:
            relevant_data_phasic = phasic_data.loc[(phasic_data.index >= min_timestamp) & (phasic_data.index < max_timestamp)]
            results.update(get_stats(np.ravel(relevant_data_phasic.values), 'phasic'))
            results.update( __compute_peaks_features(relevant_data_phasic, sampling_frequency, offset, start_WT, end_WT, thres))
            
            relevant_data_tonic = tonic_data.loc[(tonic_data.index >= min_timestamp) & (tonic_data.index < max_timestamp)]
            results.update(get_stats(np.ravel(relevant_data_tonic.values), 'tonic'))

        except Exception as e:
            pass

        return results

    else:
        return None
