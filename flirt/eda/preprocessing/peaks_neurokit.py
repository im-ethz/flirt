import numpy as np
import pandas as pd
import scipy.misc
import scipy.signal

from ..models.neurokit_misc import find_closest
from .data_utils import PeakFeatures

class ComputeNeurokitPeaks(PeakFeatures):

    """ This class computes the peak features for the relevant phasic data in the EDA signal using the Neurokit \
    peak detection algorithm. """

    def __init__(self, sampling_frequency: int = 4, amplitude_min=0.03, recovery_percentage=0.5):
        """ Construct the model to compute peak features.
        
        Parameters
        -----------
        sampling_frequency : int, optional
            the frequency with which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        amplitude_min: float, optional
            minimum threshold by which to exclude SCRs as relative to the largest amplitude in the signal
        recovery_percentage: float, optional
            percentage of total recovery time considered to compute decay time feature
        """

        self.sampling_frequency = sampling_frequency
        self.amplitude_min = amplitude_min
        self.recovery_percentage = recovery_percentage

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
        >>> peaks_features = flirt.eda.preprocessing.ComputeNeurokitPeaks().__process__(eda['eda'])
        
        References
        -----------
        - Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H., Schölzel, C., & S H Chen, A. (2020). NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing. Retrieved March 28, 2020, from https://github.com/neuropsychology/NeuroKit
        - MIT License. Copyright (c) 2020, Dominique Makowski

        - Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions: The above copyright notice and 
        this permission notice shall be included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        
        """
        
        returned_peak_data = self.__find_peaks(np.array(data.values), mean_tonic)
        
        result_df = pd.DataFrame(columns=["peaks", "amp", "max_deriv", "rise_time", "decay_time", "SCR_width", "AUC"])
        result_df['peaks'] = returned_peak_data['Peaks']
        result_df['amp'] = returned_peak_data['Amplitude']
        result_df['max_deriv'] = returned_peak_data['Max_Deriv']
        result_df['rise_time'] = returned_peak_data['RiseTime']
        result_df['decay_time'] = returned_peak_data['RecoveryTime']
        result_df['SCR_width'] = returned_peak_data['Width']

        result_df['AUC'] = result_df['amp'] * result_df['SCR_width']
        
        results = {}
        features_names = ['peaks_p', 'rise_time_p', 'max_deriv_p', 'amp_p', 'decay_time_p', 'SCR_width_p', 'auc_p_m', 'auc_p_s']

        if len(data) > 0 and len(result_df['peaks']) > 0:
            results['peaks_p'] = len(result_df['peaks'])
            results['rise_time_p'] = result_df.rise_time.mean()
            results['max_deriv_p'] = result_df.max_deriv.mean()
            results['amp_p'] = result_df.amp.mean()
            results['decay_time_p'] = result_df.decay_time.mean()
            results['SCR_width_p'] = result_df.SCR_width.mean()
            results['auc_p_m'] = result_df.AUC.mean()
            results['auc_p_s'] = result_df.AUC.sum()

            for key in features_names:
                if np.isnan(results[key]):
                    results[key] = 0.0
        else:
            for key in features_names:
                results[key] = np.nan
        
        return results

    def __find_peaks(self, signal, mean_tonic):

        """Find peaks in a signal.

        Locate peaks (local maxima) in a signal and their related characteristics, such as height (prominence),
        width and distance with other peaks.

        Parameters
        ----------
        signal : Union[list, np.array, pd.Series]
            The signal (i.e., a time series) in the form of a vector of values.
        relative_height_min : float
            The minimum height (i.e., amplitude) relative to the sample (see below). For example,
            ``relative_height_min=-2.96`` will remove all peaks which height lies below 2.96 standard
            deviations from the mean of the heights.

        Returns
        ----------
        dict
            Returns a dict itself containing 5 arrays:
            - 'Peaks' contains the peaks indices (as relative to the given signal). For instance, the
            value 3 means that the third data point of the signal is a peak.
            - 'Distance' contains, for each peak, the closest distance with another peak. Note that these
            values will be recomputed after filtering to match the selected peaks.
            - 'Height' contains the prominence of each peak. See `scipy.signal.peak_prominences()`.
            - 'Width' contains the width of each peak. See `scipy.signal.peak_widths()`.
            - 'Onset' contains the onset, start (or left trough), of each peak.
            - 'Offset' contains the offset, end (or right trough), of each peak.

        """
        signal = signal/mean_tonic

        relative_height_min=self.amplitude_min

        info = self._signal_findpeaks_scipy(signal)

        what = 'Height'
        above = relative_height_min
        keep = np.full(len(info["Peaks"]), True)
        what = info[what] / np.max(info[what])
        keep[what < above] = False

        for key in info.keys():
            info[key] = info[key][keep]
        
        # Filter
        info["Distance"] = self._signal_findpeaks_distances(info["Peaks"])
        info["Onsets"] = self._signal_findpeaks_findbase(info["Peaks"], signal, what="onset")
        info["Offsets"] = self._signal_findpeaks_findbase(info["Peaks"], signal, what="offset")
        info["Glob_Height"] = signal[info["Peaks"]]*mean_tonic

        # Peaks (remove peaks with no onset)
        valid_peaks = np.logical_and(
            info["Peaks"] > np.nanmin(info["Onsets"]), ~np.isnan(info["Onsets"])
        )  # pylint: disable=E1111
        peaks = info["Peaks"][valid_peaks]

        # Onsets (remove onsets with no peaks)
        valid_onsets = ~np.isnan(info["Onsets"])
        
        if valid_onsets.size > 1:
            valid_onsets[valid_onsets] = info["Onsets"][valid_onsets] < np.nanmax(info["Peaks"])
            onsets = info["Onsets"][valid_onsets].astype(np.int)
        else:
            onsets = np.array(info["Onsets"][valid_onsets].astype(np.int)).reshape(-1,)

        if len(onsets) != len(peaks):
            raise ValueError(
                "NeuroKit error: eda_peaks(): Peaks and onsets don't ",
                "match, so cannot get amplitude safely. Check why using `find_peaks()`.",
            )

        # Amplitude and Rise Time -------------------------------------------------

        # Amplitudes
        amplitude = np.full(len(info["Glob_Height"]), np.nan)
        amplitude[valid_peaks] = info["Glob_Height"][valid_peaks] - signal[onsets]*mean_tonic

        # Rise times
        risetime = np.full(len(info["Peaks"]), np.nan)
        risetime[valid_peaks] = (peaks - onsets) / self.sampling_frequency

        # Save info
        info["Amplitude"] = amplitude
        info["RiseTime"] = risetime

        # Recovery time -----------------------------------------------------------

        # (Half) Recovery times
        recovery = np.full(len(info["Peaks"]), np.nan)
        recovery_time = np.full(len(info["Peaks"]), np.nan)
        recovery_values = signal[onsets]*mean_tonic + (amplitude[valid_peaks] * self.recovery_percentage)

        for i, peak_index in enumerate(peaks):
            # Get segment between peak and next peak
            try:
                segment = signal[peak_index : peaks[i + 1]]*mean_tonic
            except IndexError:
                segment = signal[peak_index::]*mean_tonic

            # Adjust segment (cut when it reaches minimum to avoid picking out values on the rise of the next peak)
            segment = segment[0 : np.argmin(segment)]

            # Find recovery time
            recovery_value = find_closest(recovery_values[i], segment, direction="smaller", strictly=False)

            # Detect recovery points only if there are datapoints below recovery value
            if np.min(segment) < recovery_value:
                segment_index = np.where(segment == recovery_value)[0][0]
                recovery[np.where(valid_peaks)[0][i]] = peak_index + segment_index
                recovery_time[np.where(valid_peaks)[0][i]] = segment_index / self.sampling_frequency

        # Save ouput
        info["Recovery"] = recovery
        info["RecoveryTime"] = recovery_time

        eda_deriv = signal[1:]*mean_tonic - signal[:-1]*mean_tonic
        max_deriv = []
        for i in info['Peaks']:
            temp_start = max(0, i - self.sampling_frequency)
            max_deriv.append(max(eda_deriv[temp_start:i]))
        info["Max_Deriv"] = max_deriv

        return info
    
    def _signal_findpeaks_findbase(self, peaks, signal, what="onset"):
        
        if what == "onset":
            direction = "smaller"
        else:
            direction = "greater"

        troughs, _ = scipy.signal.find_peaks(-1 * signal)

        bases = find_closest(peaks, troughs, direction=direction, strictly=True)
        bases = np.array(bases)

        return bases
    
    def _signal_findpeaks_distances(self, peaks):

        if len(peaks) <= 2:
            distances = np.full(len(peaks), np.nan)
        else:
            distances_next = np.concatenate([[np.nan], np.abs(np.diff(peaks))])
            distances_prev = np.concatenate([np.abs(np.diff(peaks[::-1])), [np.nan]])
            distances = np.array([np.nanmin(i) for i in list(zip(distances_next, distances_prev))])

        return distances

    def _signal_findpeaks_scipy(self, signal):
        peaks, _ = scipy.signal.find_peaks(signal)

        # Get info
        distances = self._signal_findpeaks_distances(peaks)
        heights, _, __ = scipy.signal.peak_prominences(signal, peaks)
        widths, _, __, ___ = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)

        # Prepare output
        info = {"Peaks": peaks, "Distance": distances, "Height": heights, "Width": widths}

        return info
    
