import numpy as np
import pandas as pd
import scipy.signal as signal

from .data_utils import Preprocessor


class ExtendedKalmanFilter(Preprocessor):
    """
    This class filters the raw EDA data using an Extended Kalman Filter, reducing the measurement noise and eliminating motion artifacts. 
    
    Parameters
    -----------
    data : pd.Series
        raw EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`
    sampling_frequency : int, optional
        the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
    alpha : float, optional
        sudomotor nerve activation (S) response amplitude
    d_min : float, optional
        lowest rate of change for a probable sudomotor response
    d_max : float, optional
        the limit where all higher rates of changes are probably due to artifacts or noise
    min_diff : float, optional
        minimum absolute difference between two consecutive EDA measurements
    max_diff : float, optional
        maximum absolute difference between two consecutive EDA measurements
    min_Savi_Go : float, optional
        minimum Savitsky-Golay differential
    max_Savi_Go : float, optional
        maximum Savitsky-Golay differential

    Returns
    -------
    pd.Series
        series containing the filtered EDA signal using the Extended Kalman filter algorithm

    Notes
    ------
    - Decreasing `alpha` tends to filter out more of the noise and reduce the phasic amplitude response.
    - Increasing the gap between `min_diff` and `max_diff`, and between `min_Savi_Go` and `max_Savi_Go`, reduces the amount of artifact supression. 

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> import flirt.eda
    >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
    >>> eda_filtered_ekf = flirt.eda.preprocessing.ExtendedKalmanFilter(sampling_frequency=4).__process__(eda['eda'])
    
    References
    ----------
    - Tronstad C, Staal OM, Saelid S, Martinsen OG. Model-based filtering for artifact and noise suppression with state estimation for electrodermal activity measurements in real time. Annu Int Conf IEEE Eng Med Biol Soc. 2015 Aug.
    
    """

    def __init__(self, sampling_frequency: int = 4, alpha: float = 20.0, d_min: float = 0.001, d_max: float = 0.1,
                 min_diff: float = 0.0, max_diff: float = 0.05, min_Savi_Go: float = -0.01, max_Savi_Go: float = 0.01):
        self.sampling_frequency = sampling_frequency
        self.alpha = alpha
        self.d_min = d_min
        self.d_max = d_max
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.min_Savi_Go = min_Savi_Go
        self.max_Savi_Go = max_Savi_Go

    def __process__(self, data: pd.Series) -> pd.Series:
        eda = np.ravel(data)
        eda_len = len(eda)
        fs = self.sampling_frequency

        # Kalman filter (from paper: Tronstad, C.; Staal, O.M.; Saelid, S.; Martinsen, O.G. Model-based filtering for artifact and noise suppressionwith state estimation for electrodermal activity measurements in real time)
        # Constants
        alpha = self.alpha  # sudomotor nerve activation (S) response amplitude
        beta = 0.016  # magnitude of the relation between the sweat duct filling and the transport to the stratum corneum
        k_rec = 0.071  # inverse time constant governing recovery
        d_min = self.d_min  # lowest rate of change for sudomotor response
        d_max = self.d_max  # all higher rates of change probably due to artifacts or noise
        cov_SCR_S = 0.01  # cavariance between SCR and S
        var_S = 0.01  # Variance of S
        var_SCR = 0.001  # Variance of SCR
        var_k_diff = 0.01  # Variance of k_diff
        var_SC_0 = 0.01  # Variance of SC_0
        var_SC_h = 0.01  # Variance of SC_h
        k_R = 100  # noise parameter time constant
        R_0 = 5  # noise parameter constant
        # non-zero K if absolute difference between two consecutive measurements is between min_diff
        # and max_diff, and Savitsky-Golay differential is between min_Savi_Go and max_Savi_Go
        # CHANGE THESE TO CONTROL AMOUNT OF SMOOTHING
        min_diff = self.min_diff
        max_diff = self.max_diff
        min_Savi_Go = self.min_Savi_Go
        max_Savi_Go = self.max_Savi_Go

        ##### EDA model
        # state, x = [SC_H, k_diff, SC_0, SCR, S]'
        # Discretise continuous system of equations (delta T = 0.25 as data sampled at 4Hz)
        # Linearise non-linear system around the last predicted state in each time-step
        # x(k+1) = h(x(k), u(k)) = dh/dx * x(k) + dh/du * u(k)
        # Measurement model: y = SC = Hx 
        x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        P_0 = np.identity(5) * 0.25
        Q = np.array(
            [[var_SC_h, 0, 0, 0, 0],
             [0, var_k_diff, 0, 0, 0],
             [0, 0, var_SC_0, 0, 0],
             [0, 0, 0, var_SCR, cov_SCR_S],
             [0, 0, 0, cov_SCR_S, var_S]])
        delta_T = 1 / fs

        # Initialisation
        x_m = x_0
        P_m = P_0
        x_p = np.zeros_like(x_m)
        P_p = P_m
        H = np.array([1, 0, 1, 1, 0])
        H = H
        k = 0
        count = 0
        SC = np.zeros((eda_len,))
        SCR = np.zeros((eda_len,))
        SC_0 = np.zeros((eda_len,))

        # Recursion
        while k <= eda_len - 2:
            k = k + 1
            ### Step 1: x_p = h(x_m, U), P_p = A*P_m*A' where A = dh(x_m, U)/dx 
            if k + 1 < (2 * 4 - 1):
                if k + 1 <= 2:
                    dSC = signal.savgol_filter(eda[0:k + 1], window_length=2 * k - 1, polyorder=k - 1, deriv=1,
                                               delta=1.0, axis=- 1, mode='interp', cval=0.0)
                else:
                    dSC = signal.savgol_filter(eda[0:k + 1], window_length=3, polyorder=2, deriv=1, delta=1.0, axis=- 1,
                                               mode='interp', cval=0.0)
            else:
                dSC = signal.savgol_filter(eda[0:k + 1], window_length=2 * 4 - 1, polyorder=2, deriv=1, delta=1.0,
                                           axis=- 1, mode='interp', cval=0.0)

            if dSC[k] < d_min or dSC[k] > d_max:
                U = 0
            else:
                U = 1

            x_p[0,] = x_m[0,] + delta_T * (beta * x_m[3,] - x_m[1,] * x_m[0,])
            x_p[1,] = x_m[1,]
            x_p[2,] = x_m[2,]
            x_p[3,] = x_m[3,] + delta_T * (alpha * x_m[4,] - k_rec * x_m[3,])
            x_p[4,] = (U - x_m[4,]) * x_m[4,]

            A = np.array(
                [[1 - delta_T * x_m[1], -delta_T * x_m[0], 0, delta_T * beta, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                 [0, 0, 0, 1 - delta_T * k_rec, delta_T * alpha], [0, 0, 0, 0, U - 2 * x_m[4]]])
            P_p = A.dot(Q).dot(A.transpose())
            w, v = np.linalg.eig(A)

            for i in w:
                AI = A - i * np.identity(5)
                AIH = np.concatenate((AI, H.reshape(1, 5)), axis=0)
                rank = np.linalg.matrix_rank(AIH)

            # Step 2: K = P_p*H'*(H*P_p*H'+ R)^-1, x_m = x_p + K*(z - y), P_m = (I-K*H)*P_p
            y = H.dot(x_p)
            eda_diff = abs(eda[k] - eda[k - 1])  # Difference between consecutive measurements of EDA
            if eda_diff >= min_diff and eda_diff <= max_diff and dSC[k] >= min_Savi_Go and dSC[k] <= max_Savi_Go:
                if k + 1 < 10 * 4:
                    noise_est = np.std(dSC[0:k + 1])  # std of differential of all measurements over last few sec
                else:
                    noise_est = np.std(
                        dSC[k - 10 * 4 + 1:k + 1])  # std of differential of all measurements over last 10 sec
                R = k_R * noise_est + R_0
                K = P_p.dot(H.transpose()) * (1 / (H.dot(P_p).dot(H.transpose()) + R))
                count = count + 1
            else:
                K = np.zeros((5,))

            x_m = x_p + K * (eda[k] - y)
            # P_m = (np.identity(5) - K.dot(H)).dot(P_p)

            if np.isnan(x_m[0]):
                break

            SC[k,] = x_m[0,] + x_m[2,] + x_m[3,]

        # print('- Artifact removal and noise filtering using the EKF completed.')

        # Return SC filtered data as a pd.Series
        return pd.Series(data=SC[:eda_len], index=data.index, name='filtered_eda')
