from __future__ import division
import numpy as np
import pandas as pd

from .data_utils import SignalDecomposition
from .low_pass import LowPassFilter
from ..models.ledalab import leda2, utils, analyse, deconvolution

class LedaLab(SignalDecomposition):
    """ This class decomposes the filtered EDA signal into tonic and phasic components using the Ledalab algorithm, adapted from MATLAB. It also \
    ensures that the tonic and phasic signals never drop below zero."""

    def __init__(self, sampling_rate: int=4, downsample: int=1, optimisation: int=0):
        """ Construct the signal decomposition model.

        Parameters
        ----------- 
        sampling_rate : int, optional
            the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        downsample: int, optional
            downsampling factor to reduce computing time (1 == no downsample)
        optimisation: int, optional
            level of optimization (0 == no optimisation) to find the optimal parameter tau of the bateman equation using gradient descent   

        """
        
        self.sampling_rate = sampling_rate
        self.downsample = downsample
        self.optimisation = optimisation

    def __process__(self, data: pd.Series) -> (pd.Series, pd.Series):
        """Decompose electrodermal activity into phasic and tonic components.

        Parameters
        -----------
        data : pd.Series
            filtered EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), \
            columns is the filtered eda data: `eda`
        
        Returns
        -------
        pd.Series, pd.Series
            two dataframes containing the phasic and the tonic components, index is a list of \
                timestamps for both dataframes

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> phasic, tonic = flirt.eda.preprocessing.LedaLab().__process__(eda['eda'])
        
        References
        ----------
        - Benedek, M. & Kaernbach, C. Decomposition of skin conductance data by means of nonnegative deconvolution. Psychophysiology. 2010
        - https://github.com/HIIT/Ledapy
        """

        leda2.reset()
        leda2.current.do_optimize = self.optimisation
        self.import_data(data)
        deconvolution.sdeco(self.optimisation)
        
        phasic = leda2.analysis.phasicData
        tonic = leda2.analysis.tonicData

        # Creating dataframe
        data_phasic = pd.Series(np.ravel(phasic), data.index)
        data_tonic = pd.Series(np.ravel(tonic), data.index)

        
        # Check if tonic values are below zero and filter if it is the case
        if np.amin(data_tonic.values) < 0:
            lowpass_filter = LowPassFilter(sampling_frequency=self.sampling_rate, order=1, cutoff=0.2, filter='butter')
            filtered_tonic = lowpass_filter.__process__(data_tonic)
            data_tonic = filtered_tonic

        # Check if phasic values are below zero and filter if it is the case
        if np.amin(data_phasic.values) < 0:
            lowpass_filter = LowPassFilter(sampling_frequency=self.sampling_rate, order=1, cutoff=0.25, filter='butter')
            filtered_phasic = lowpass_filter.__process__(data_phasic)
            data_phasic = filtered_phasic
        

        return data_phasic, data_tonic

    def import_data(self, data):
        """
        Sets leda2 object to its appropriate values to allow analysis
        Adapted from main/import/import_data.m
        """
        
        conductance_data = np.array(data.values, dtype='float64')
        time_data = utils.genTimeVector(conductance_data, self.sampling_rate)

        if self.downsample > 1:
            (time_data, conductance_data) = utils.downsamp(time_data, conductance_data, self.downsample, 'mean')
        leda2.data.samplingrate = self.sampling_rate / self.downsample
        leda2.data.time_data = time_data
        leda2.data.conductance_data = conductance_data
        leda2.data.conductance_error = np.sqrt(np.mean(pow(np.diff(conductance_data), 2)) / 2)
        leda2.data.N = len(conductance_data)
        leda2.data.conductance_min = np.min(conductance_data)
        leda2.data.conductance_max = np.max(conductance_data)
        leda2.data.conductance_smoothData = conductance_data
        analyse.trough2peak_analysis()

    
