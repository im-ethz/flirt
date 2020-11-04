import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from .preprocessing import data_utils, CvxEda, LowPassFilter, ComputePeaks, LedaLab, ExtendedKalmanFilter, MitExplorerDetector, MultiStepPipeline, LrDetector
from ..stats.common import get_stats



def get_eda_features(data: pd.Series, data_frequency: int = 4, window_length: int = 60, window_step_size: int = 1, num_cores=0,
                     preprocessor: data_utils.Preprocessor = LowPassFilter(),
                     signal_decomposition: data_utils.SignalDecomposition = CvxEda(),
                     scr_features: data_utils.PeakFeatures = ComputePeaks()):
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
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available
    preprocessor: class, optional
        the method chosen to clean the data: low-pass filtering or Kalman filtering or artifact-detection with low-pass filtering
    signal_decomposition: class, optional
        the method chosen to decompose the data (default: CvxEda)
    scr_features: class, optional
        computes peak features of the Skin Conductance response data using the algorithm from MIT Media Lab

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
    >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
    >>> eda_features = flirt.eda.get_eda_features(eda, data_frequency=4, window_length=60, window_step_size=1)
    """

    # Get number of cores
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    # Filter Data
    filtered_dataframe = preprocessor.__process__(data)

    # Decompose Data
    phasic_data, tonic_data = signal_decomposition.__process__(filtered_dataframe)

    # advance by window_step_size * data_frequency
    inputs = trange(0, len(data) - 1, window_step_size * data_frequency, desc="EDA features")
    
    # Get features
    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(delayed(__get_features_per_window)(phasic_data, tonic_data, window_length=window_length,
                                                              i=k, sampling_frequency=data_frequency, __compute_peak_features = scr_features) for k in inputs)

    # results = pd.DataFrame(list(filter(None, results)))
    results = pd.DataFrame(list(results))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)


    return results


def __get_features_per_window(phasic_data: pd.Series, tonic_data: pd.Series, window_length: int, i: int,
                              sampling_frequency: int = 4,
                               __compute_peak_features: data_utils.PeakFeatures = ComputePeaks()):

    if pd.Timedelta(phasic_data.index[i + 1] - phasic_data.index[i]).total_seconds() <= window_length:
        min_timestamp = phasic_data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)

        results = {
            'datetime': max_timestamp,
        }

        try:
            relevant_data_phasic = phasic_data.loc[
                (phasic_data.index >= min_timestamp) & (phasic_data.index < max_timestamp)]
            results.update(get_stats(np.ravel(relevant_data_phasic.values), 'phasic'))
            results.update(
                __compute_peak_features.__process__(relevant_data_phasic))

            relevant_data_tonic = tonic_data.loc[
                (tonic_data.index >= min_timestamp) & (tonic_data.index < max_timestamp)]
            results.update(get_stats(np.ravel(relevant_data_tonic.values), 'tonic'))
    
        except Exception as e:
            pass

        return results

    else:
        return None

