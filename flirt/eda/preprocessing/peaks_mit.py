import numpy as np
import pandas as pd

from .data_utils import PeakFeatures

class ComputeMITPeaks(PeakFeatures):

    """ This class computes the peak features for the relevant phasic data in the EDA signal using the EDAexplorer \
    peak detection algorithm from MIT. """         
    
    def __init__(self, sampling_frequency: int = 4, offset: int = 1, start_WT: int = 3, end_WT: int = 10,
                 thres: float = 0.01):
        """ Construct the model to compute peak features.
        
        Parameters
        -----------
        sampling_frequency : int, optional
            the frequency with which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        offset: int, optional
            the number of rising samples and falling samples after a peak needed to be counted as a peak
        start_WT: int, optional
            maximum number of seconds before the apex of a peak that is the "start" of the peak
        end_WT: int, optional
            maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
        thres: float, optional
            the minimum uS change required to register as a peak (0 means all peaks count)
        """

        self.sampling_frequency = sampling_frequency
        self.offset = offset
        self.start_WT = start_WT
        self.end_WT = end_WT
        self.thres = thres

    def __process__(self, data: pd.Series, mean_tonic: float) -> dict:
        """ Perform the peak detection and compute the relevant features.

        Parameters
        ----------
        data : pd.Series
            relevant data onto which compute peak features , index is a list of timestamps according to the sampling frequency (e.g. 4Hz for Empatica)

        Returns
        -------
        dict
            dictionary containing the peak features computed on the phasic component of the EDA data (peaks_p: number of peaks, rise_time_p: average rise time of the peaks, \
            max_deriv_p: average value of the maximum derivative, amp_p: average amplitute of the peaks, decay_time_p: average decay time of the peaks, SCR_width_p: average width of the peak (SCR), \
            auc_p: average area under the peak, auc_s: total area under all peaks)
        
        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda.preprocessing
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> peaks_features = flirt.eda.preprocessing.ComputeMITPeaks().__process__(eda['eda'])
        
        References
        -----------
        - Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data. In Engineering in Medicine and Biology Conference. 2015.
        - https://github.com/MITMediaLabAffectiveComputing/eda-explorer
        """

        returned_peak_data = self.__find_peaks(data, mean_tonic)
        result_df = pd.DataFrame(columns=["peaks", "amp", "max_deriv", "rise_time", "decay_time", "SCR_width"])
        result_df['peaks'] = returned_peak_data[0]
        result_df['amp'] = returned_peak_data[5]
        result_df['max_deriv'] = returned_peak_data[6]
        result_df['rise_time'] = returned_peak_data[7]
        result_df['decay_time'] = returned_peak_data[8]
        result_df['SCR_width'] = returned_peak_data[9]
        result_df['start_peak'] = returned_peak_data[2]

        # To keep all filtered data remove this line
        feature_data = result_df[result_df.peaks == 1][
            ['peaks', 'rise_time', 'max_deriv', 'amp', 'decay_time', 'SCR_width', 'start_peak']]

        # Replace 0s with NaN, this is where the 50% of the peak was not found, too close to the next peak
        # feature_data[['SCR_width','decay_time']]=feature_data[['SCR_width','decay_time']].replace(0, np.nan)
        feature_data['AUC'] = feature_data['amp'] * feature_data['SCR_width']
        #feature_data['AUC_precise'] = abs(np.diff(data[data.index[feature_data['start_peak']]:data.index[feature_data['start_peak'] + timedelta(seconds=5.0)]])).sum()

        results = {}
        features_names = ['peaks_p', 'rise_time_p', 'max_deriv_p', 'amp_p', 'decay_time_p', 'SCR_width_p', 'auc_p_m', 'auc_p_s']
        if len(data) > 0:
            results['peaks_p'] = len(feature_data)
            results['rise_time_p'] = result_df[result_df.peaks != 0.0].rise_time.mean()
            results['max_deriv_p'] = result_df[result_df.peaks != 0.0].max_deriv.mean()
            results['amp_p'] = result_df[result_df.peaks != 0.0].amp.mean()
            results['decay_time_p'] = feature_data[feature_data.peaks != 0.0].decay_time.mean()
            results['SCR_width_p'] = feature_data[feature_data.peaks != 0.0].SCR_width.mean()
            results['auc_p_m'] = feature_data[feature_data.peaks != 0.0].AUC.mean()
            results['auc_p_s'] = feature_data[feature_data.peaks != 0.0].AUC.sum()
            for key in features_names:
                if np.isnan(results[key]):
                    results[key] = 0.0
        else:
            for key in features_names:
                results[key] = np.nan

        return results

    def __find_peaks(self, data, mean_tonic):
        """
        This function finds the peaks of an EDA signal and returns basic properties.
        Also, peak_end is assumed to be no later than the start of the next peak.

        ********* INPUTS **********
        data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
        offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak
        start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
        end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
        thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
        sampleRate:  number of samples per second, default=8

        ********* OUTPUTS **********
        peaks:               list of binary, 1 if apex of SCR
        peak_start:          list of binary, 1 if start of SCR
        peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
        peak_end:            list of binary, 1 if rec.t/2 of SCR
        peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
        amplitude:           list of floats,  value of EDA at apex - value of EDA at start
        max_deriv:           list of floats, max derivative within 1 second of apex of SCR
        """
        data = data/mean_tonic # normalise SCR for global input parameters
        sample_rate = self.sampling_frequency
        offset = self.offset*self.sampling_frequency
        start_WT = self.start_WT
        end_WT = self.end_WT
        thres = self.thres

        eda_deriv = data[1:].values - data[:-1].values
        peaks = np.zeros(len(eda_deriv))
        peak_sign = np.sign(eda_deriv)
        for i in range(int(offset), int(len(eda_deriv) - offset)):
            if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
                peaks[i] = 1
                for j in range(1, int(offset)):
                    if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                        peaks[i] = 0
                        break

        # Finding start of peaks
        peak_start = np.zeros(len(eda_deriv))
        peak_start_times = [''] * len(data)
        max_deriv = np.zeros(len(data))
        rise_time = np.zeros(len(data))

        for i in range(0, len(peaks)):
            if peaks[i] == 1:
                temp_start = max(0, i - sample_rate)
                max_deriv[i] = max(eda_deriv[temp_start:i])
                start_deriv = .01 * max_deriv[i]

                found = False
                find_start = i
                # has to peak within start_WT seconds
                while not found and find_start > (i - start_WT * sample_rate):
                    if eda_deriv[find_start] < start_deriv:
                        found = True
                        peak_start[find_start] = 1
                        peak_start_times[i] = data.index[find_start]
                        rise_time[i] = self.__get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))

                    find_start = find_start - 1

                # If we didn't find a start
                if not found:
                    peak_start[i - start_WT * sample_rate] = 1
                    peak_start_times[i] = data.index[i - start_WT * sample_rate]
                    rise_time[i] = start_WT

                # Check if amplitude is too small
                if thres > 0 and (data.iloc[i] - data[peak_start_times[i]]) < thres:
                    peaks[i] = 0
                    peak_start[i] = 0
                    peak_start_times[i] = ''
                    max_deriv[i] = 0
                    rise_time[i] = 0

        # Finding the end of the peak, amplitude of peak
        peak_end = np.zeros(len(data))
        peak_end_times = [''] * len(data)
        amplitude = np.zeros(len(data))
        decay_time = np.zeros(len(data))
        half_rise = [''] * len(data)
        scr_width = np.zeros(len(data))

        for i in range(0, len(peaks)):
            if peaks[i] == 1:
                peak_amp = data.iloc[i]
                start_amp = data[peak_start_times[i]]
                amplitude[i] = peak_amp - start_amp

                half_amp = amplitude[i] * .5 + start_amp

                found = False
                find_end = i
                # has to decay within end_WT seconds
                while found == False and find_end < (i + end_WT * sample_rate) and find_end < len(peaks):
                    if data.iloc[find_end] < half_amp:
                        found = True
                        peak_end[find_end] = 1
                        peak_end_times[i] = data.index[find_end]
                        decay_time[i] = self.__get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])

                        # Find width
                        find_rise = i
                        found_rise = False
                        while not found_rise:
                            if data.iloc[find_rise] < half_amp:
                                found_rise = True
                                half_rise[i] = data.index[find_rise]
                                scr_width[i] = self.__get_seconds_and_microseconds(
                                    pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                            find_rise = find_rise - 1

                    elif peak_start[find_end] == 1:
                        found = True
                        peak_end[find_end] = 1
                        peak_end_times[i] = data.index[find_end]
                    find_end = find_end + 1

                # If we didn't find an end
                if not found:
                    min_index = np.argmin(data.iloc[i:(i + end_WT * sample_rate)].tolist())
                    peak_end[i + min_index] = 1
                    peak_end_times[i] = data.index[i + min_index]

        peaks = np.concatenate((peaks, np.array([0])))
        peak_start = np.concatenate((peak_start, np.array([0])))
        max_deriv = max_deriv * sample_rate  # now in change in amplitude over change in time form (uS/second)

        return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude*mean_tonic, max_deriv*mean_tonic, rise_time, decay_time, scr_width, half_rise

    def __get_seconds_and_microseconds(self, pandas_time):
        return pandas_time.seconds + pandas_time.microseconds * 1e-6
