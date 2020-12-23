import multiprocessing
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from .preprocessing import data_utils, CvxEda, LowPassFilter, ComputeMITPeaks, ComputeNeurokitPeaks, LedaLab, ExtendedKalmanFilter, MitExplorerDetector, MultiStepPipeline, LrDetector
from ..stats.common import get_stats
from .preprocessing import get_MFCC_stats, get_fd_stats



def get_eda_features(data: pd.Series, data_frequency: int = 4, discard: int = 0, window_length: int = 60, window_step_size: float = 1.0, num_cores=2,
                     preprocessor: data_utils.Preprocessor = LowPassFilter(),
                     signal_decomposition: data_utils.SignalDecomposition = CvxEda(),
                     scr_features: data_utils.PeakFeatures = ComputeMITPeaks()):
    """
    This function computes several time-domain and frequency-based features for the phasic and tonic components of the EDA signal.
    
    Parameters
    ----------
    data : pd.Series
        raw EDA data , index is a list of timestamps according to the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`
    data_frequency : int, optional
        the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
    discard: int, optional
        number of seconds of data to discard from the filtered data onto which to compute the SCR/SCL decomposition
    window_length: float, optional
        length of window to slide over the data, to compute features (in seconds)
    window_step_size: float, optional
        step-size taken to move the sliding window (in seconds)
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available
    preprocessor: class, optional
        the method chosen to clean the data: low-pass filtering, Kalman filtering, particle filtering, artifact-detection with low-pass filtering (default: low-pass filtering)
    signal_decomposition: class, optional
        the method chosen to decompose the data: CvxEDA or LedaLab (default: CvxEda)
    scr_features: class, optional
        computes peak features of the Skin Conductance response data using the algorithm from MIT or that from Neurokit (default: MIT)

    Returns
    -------
    pd.DataFrame
        dataframe containing all time-domain, peak and frequency-domain features, index is a list of timestamps according to the window step size

    Notes
    -----
    The output dataframe contains the following EDA features computed for both the tonic and phasic components

        - **Time-Domain Features**: mean, std, min, max, ptp, sum, energy, skewness, kurtosis, \
        peaks, rms, lineintegral, n_above_mean, n_below_mean, n_sign_changes, iqr, iqr_5_95, pct_5, pct_95, \
        entropy, perm_entropy, svd_entropy
        - **Peak Features (only on SCR)**: peaks_p, rise_time_p, max_deriv_p, amp_p, decay_time_p, SCR_width_p, auc_p_m, auc_p_s
        - **Frequency-Domain Features**: fd_sma, fd_energy, fd_varPower, fd_Power_0.05, fd_Power_0.15, fd_Power_0.25, fd_Power_0.35, \
        fd_Power_0.45, fd_kurtosis, fd_iqr, 
        - **Time-Frequency-Domain Features**: mfcc_mean, mfcc_std, mfcc_median, mfcc_skewness, mfcc_kurtosis, mfcc_iqr

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
    phasic_data, tonic_data = signal_decomposition.__process__(filtered_dataframe[discard*data_frequency:])

    # advance by window_step_size * data_frequency
    inputs = trange(0, len(data) - 1, 
            int(window_step_size * data_frequency), desc="EDA features")
            

    # Get features
    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(delayed(__get_features_per_window)(phasic_data, tonic_data, window_length=window_length,
                                                              i=k, sampling_frequency=data_frequency, __compute_peak_features = scr_features) for k in inputs)

    # results = pd.DataFrame(list(filter(None, results)))
    results = pd.DataFrame(list(results))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)
    print(results)

    return results


def __get_features_per_window(phasic_data: pd.Series, tonic_data: pd.Series, window_length: int, i: int,
                              sampling_frequency: int = 4,
                               __compute_peak_features: data_utils.PeakFeatures = ComputeMITPeaks()):

    if pd.Timedelta(phasic_data.index[i + 1] - phasic_data.index[i]).total_seconds() <= window_length:
        min_timestamp = phasic_data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)

        results = {
            'datetime': max_timestamp,
        }

        try:
            relevant_data_phasic = phasic_data.loc[
                (phasic_data.index >= min_timestamp) & (phasic_data.index < max_timestamp)]
            relevant_data_tonic = tonic_data.loc[
                (tonic_data.index >= min_timestamp) & (tonic_data.index < max_timestamp)]
            mean_tonic = relevant_data_tonic.mean()
            
            results.update(get_stats(np.ravel(relevant_data_phasic.values), 'phasic'))
            results.update(get_fd_stats(np.ravel(relevant_data_phasic.values), 'phasic'))
            results.update(get_MFCC_stats(np.ravel(relevant_data_phasic.values), 'phasic'))
            results.update(
                __compute_peak_features.__process__(relevant_data_phasic, mean_tonic))

            results.update(get_stats(np.ravel(relevant_data_tonic.values), 'tonic'))
            results.update(get_fd_stats(np.ravel(relevant_data_tonic.values), 'tonic'))
            results.update(get_MFCC_stats(np.ravel(relevant_data_tonic.values), 'tonic'))
    
        except Exception as e:
            pass

        return results

    else:
        return None

