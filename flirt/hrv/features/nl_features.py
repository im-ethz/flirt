import nolds as nd
import numpy as np

from flirt.hrv.features.data_utils import DomainFeatures


class NonLinearFeatures(DomainFeatures):
    def __init__(self, emb_dim: int = 2):
        self.emb_dim = emb_dim

    def __get_type__(self) -> str:
        return "Non-Linear"

    def __generate__(self, data: np.array) -> dict:
        data_np = np.asarray(data)

        results = {
            # 'hrv_DFA_alpha1': np.nan,
            # 'hrv_DFA_alpha2': np.nan,
            # 'hrv_sample_entropy_nk2': np.nan
            # 'hrv_correlation_dimension': np.nan,
            'hrv_SD1': np.nan,
            'hrv_SD2': np.nan,
            'hrv_SD2SD1': np.nan,
            'hrv_CSI': np.nan,
            'hrv_CVI': np.nan,
            'hrv_CSI_Modified': np.nan,
        }

        if len(data_np) > self.emb_dim:
            # This is the sample entropy specifically calculated for HRV based on neurokit2!
            # results['hrv_sample_entropy_nk2'] = nk.entropy_sample(data_np, dimension=emb_dim, r=0.2 * np.std(data_np, ddof=1))
            # results['hrv_correlation_dimension'] = __correlation_dimension(data_np, emb_dim)
            results.update(_nonlinear(data_np))

            # very compute intensive
            # if len(data_np) > 17:
            #    results['hrv_DFA_alpha1'] = __detrend_fluctuation_analysis_alpha_1(data_np)

            # if len(data_np) > 66:
            #    results['hrv_DFA_alpha2'] = __detrend_fluctuation_analysis_alpha_2(data_np)

        # TODO: Raise one error in case DFA features are not possible to compute not everytime the function is called...

        return results


def _detrend_fluctuation_analysis_alpha_1(data):
    alpha_1 = nd.dfa(data, range(4, 16), fit_exp="poly")
    return alpha_1


def _detrend_fluctuation_analysis_alpha_2(data):
    alpha_2 = nd.dfa(data, range(16, 64), fit_exp="poly")
    return alpha_2


def _correlation_dimension(data, emb_dim):
    emb_dim = nd.corr_dim(data, emb_dim=emb_dim, fit="poly")
    return emb_dim


def _nonlinear(rri):
    diff_rri = np.diff(rri)
    out = {}  # Initialize empty container for results

    # Poincaré
    sd_rri = np.std(rri, ddof=1) ** 2
    sd_heart_period = np.std(diff_rri, ddof=1) ** 2
    out["hrv_SD1"] = np.sqrt(sd_heart_period * 0.5)
    out["hrv_SD2"] = np.sqrt(2 * sd_rri - 0.5 * sd_heart_period)
    out["hrv_SD2SD1"] = out["hrv_SD2"] / out["hrv_SD1"]

    # CSI / CVI
    T = 4 * out["hrv_SD1"]
    L = 4 * out["hrv_SD2"]
    out["hrv_CSI"] = L / T
    out["hrv_CVI"] = np.log10(L * T)
    out["hrv_CSI_Modified"] = L ** 2 / T

    return out
