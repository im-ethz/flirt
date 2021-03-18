from collections import namedtuple
from typing import List

import numpy as np
from astropy.timeseries import LombScargle
from scipy import interpolate
from scipy import signal

from flirt.hrv.features.data_utils import DomainFeatures

VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])


class FdFeatures(DomainFeatures):
    def __init__(self, sampling_frequency: int = 1, method: str = 'lomb'):
        self.sampling_frequency = sampling_frequency
        self.method = method

    def __get_type__(self) -> str:
        return "Frequency Domain"

    def __generate__(self, data: np.array) -> dict:
        return get_fd_features(data, method=self.method, sampling_frequency=self.sampling_frequency)


def get_fd_features(data, method: str = "lomb", sampling_frequency: int = 4, interpolation_method: str = "linear",
                    vlf_band: namedtuple = VlfBand(0.003, 0.04),
                    lf_band: namedtuple = LfBand(0.04, 0.15),
                    hf_band: namedtuple = HfBand(0.15, 0.40)):
    data_np = np.asarray(data)

    results = {
        'hrv_total_power': np.nan,
        'hrv_vlf': np.nan,
        'hrv_lf': np.nan,
        'hrv_hf': np.nan,
        'hrv_lf_hf_ratio': np.nan,
        'hrv_lfnu': np.nan,
        'hrv_hfnu': np.nan,
    }

    results.update(__frequency_domain(nn_intervals=data_np, method=method, sampling_frequency=sampling_frequency,
                                      interpolation_method=interpolation_method, vlf_band=vlf_band, lf_band=lf_band,
                                      hf_band=hf_band))

    return results


def __frequency_domain(nn_intervals, method, sampling_frequency, interpolation_method, vlf_band, lf_band, hf_band):
    freq, psd = __get_freq_psd_from_nn_intervals(nn_intervals, method, sampling_frequency, interpolation_method,
                                                 vlf_band, hf_band)

    frequency_domain_features = __get_features_from_psd(freq=freq, psd=psd, vlf_band=vlf_band, lf_band=lf_band,
                                                        hf_band=hf_band)

    return frequency_domain_features


def __get_freq_psd_from_nn_intervals(nn_intervals, method, sampling_frequency, interpolation_method, vlf_band, hf_band):
    timestamp_list = __create_timestamp_list(nn_intervals)

    if method == "welch":
        funct = interpolate.interp1d(x=timestamp_list, y=nn_intervals, kind=interpolation_method)
        timestamps_interpolation = __create_interpolated_timestamp_list(nn_intervals, sampling_frequency)
        nni_interpolation = funct(timestamps_interpolation)
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)
        freq, psd = signal.welch(x=nni_normalized, fs=sampling_frequency, window='hann', nfft=4096)

    elif method == "lomb":
        freq, psd = LombScargle(timestamp_list, nn_intervals, normalization='psd').autopower(
            minimum_frequency=vlf_band[0],
            maximum_frequency=hf_band[1])

    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq, psd


def __create_timestamp_list(nn_intervals: List[float]):
    nni_tmstp = np.cumsum(nn_intervals) / 1000
    return nni_tmstp - nni_tmstp[0]


def __create_interpolated_timestamp_list(nn_intervals: List[float], sampling_frequency: int = 7):
    time_nni = __create_timestamp_list(nn_intervals)
    nni_interpolation_tmstp = np.arange(0, time_nni[-1], 1 / float(sampling_frequency))
    return nni_interpolation_tmstp


def __get_features_from_psd(freq: List[float], psd: List[float], vlf_band, lf_band, hf_band):
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    # Integrate using the composite trapezoidal rule
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

    # total power & vlf : Feature often used for  "long term recordings" analysis
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    total_power = vlf + lf + hf

    if lf > 0 and hf > 0:
        lf_hf_ratio = lf / hf
    else:
        lf_hf_ratio = np.nan

    if lf > 0 and hf > 0:
        lfnu = (lf / (lf + hf)) * 100
        hfnu = (hf / (lf + hf)) * 100
    else:
        lfnu = np.nan
        hfnu = np.nan

    freqency_domain_features = {
        'hrv_lf': lf,
        'hrv_hf': hf,
        'hrv_lf_hf_ratio': lf_hf_ratio,
        'hrv_lfnu': lfnu,
        'hrv_hfnu': hfnu,
        'hrv_total_power': total_power,
        'hrv_vlf': vlf
    }

    return freqency_domain_features
