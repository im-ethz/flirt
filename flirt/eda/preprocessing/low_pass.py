import numpy as np
import pandas as pd
from scipy.signal import iirfilter, lfilter

from .data_utils import Preprocessor


class LowPassFilter(Preprocessor):
    """ This class filters the raw EDA data using a infinite impulse response (IIR) low-pass filter to remove high frequency noise. 
    """

    def __init__(self, sampling_frequency: int = 4, order: int = 1, cutoff: float = 0.1,
                 filter: str = 'butter', rp: int = 1, rs: int = 5):
        """ Construct the low-pass filtering model.

        Parameters
        -----------
        sampling_frequency : int
            the frequency with which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        order : int, optional
            order of the filter
        cutoff : float, optional
            desired cutoff frequency, below which the signal is kept (can be approximated using spectral analysis and determining the most prominent frequencies to keep)
        filter : str, optional
            the type of filter to use: 'butter' for Butterworth filter, 'cheby1' for Chebyshev I filter, 'cheby2' for Chebyshev II filter, 'ellip' for Cauer filter, 'bessel' for Bessel filter
        rp : int, optional
            the maximum ripple in the passband, in dB (for Chebyshev and elliptic filters only)
        rs : ints, optional
            the minimum attenuation in the stop band, in dB (for Chebyshev and elliptic filters only)
        """

        self.sampling_frequency = sampling_frequency
        self.order = order
        self.cutoff = cutoff
        self.filter = filter
        self.rp = rp
        self.rs = rs

    def __process__(self, data: pd.Series) -> pd.Series:
        """ Perform the signal filtering.

        Parameters
        ----------
        data : pd.Series
            EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the eda data to be filtered

        Returns
        -------
        pd.Series
            dataframe containing the filtered EDA signal using the low-pass filter of choice

        Notes
        ------
        - Decreasing the cutoff frequency allows for more noise filtering. An appropriate cutoff frequency is essential to remove enough noise without distorting the signal.  

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda.preprocessing
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> eda_filtered_low_pass = flirt.eda.preprocessing.LowPassFilter(sampling_frequency=4, order=1, cutoff=0.1, \
            filter='butter').__process__(eda['eda'])
        
        References
        ----------
        - https://docs.scipy.org/doc/scipy/reference/signal.html
        """

        # Load EDA data
        # Use loaded data eg from flirt.reader.empatica.read_eda_file_into_df(filename)
        eda = data
        datetime = eda.index
        eda_len = len(eda)
        eda = np.array(eda).reshape(eda_len)

        ## IMPLEMENT MULTIPLE IIR FILTERS (BUTERWORTH, CHEBYSHEV 1,2, CAUER, BESSEL)
        normal_cutoff = self.cutoff / (0.5 * self.sampling_frequency)

        # a,b filtering
        b, a = iirfilter(self.order, normal_cutoff, rp=self.rp, rs=self.rs, btype='lowpass', analog=False,
                         ftype=self.filter, fs=self.sampling_frequency, output='ba')
        eda_filtered = lfilter(b, a, eda)

        # Creating dataframe
        eda_filtered_dataframe = pd.Series(eda_filtered[:eda_len], index=datetime[:eda_len])

        return eda_filtered_dataframe
