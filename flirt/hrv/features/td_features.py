import numpy as np

from flirt.hrv.features.data_utils import DomainFeatures


class TdFeatures(DomainFeatures):
    def __get_type__(self) -> str:
        return "Time Domain"

    def __generate__(self, data: np.array) -> dict:
        nn_intervals = np.asarray(data)

        out = {}  # Initialize empty container for results

        diff_nni = np.diff(nn_intervals)
        length_int = len(nn_intervals)
        hr = np.divide(60000, nn_intervals)

        nni_50 = sum(np.abs(diff_nni) > 50)
        nni_20 = sum(np.abs(diff_nni) > 20)
        mean_nni = np.mean(nn_intervals)
        median_nni = np.median(nn_intervals)
        rmssd = np.sqrt(np.mean(diff_nni ** 2))
        sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1

        out['hrv_mean_nni'] = mean_nni
        out['hrv_median_nni'] = median_nni
        out['hrv_range_nni'] = max(nn_intervals) - min(nn_intervals)
        out['hrv_sdsd'] = np.std(diff_nni)
        out['hrv_rmssd'] = rmssd
        out['hrv_nni_50'] = nni_50
        out['hrv_pnni_50'] = 100 * nni_50 / length_int
        out['hrv_nni_20'] = nni_20
        out['hrv_pnni_20'] = 100 * nni_20 / length_int
        out['hrv_cvsd'] = rmssd / mean_nni
        out['hrv_sdnn'] = sdnn
        out['hrv_cvnni'] = sdnn / mean_nni
        out['hrv_mean_hr'] = np.mean(hr)
        out['hrv_min_hr'] = min(hr)
        out['hrv_max_hr'] = max(hr)
        out['hrv_std_hr'] = np.std(hr)

        return out
