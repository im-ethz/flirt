import pandas as pd

from .artefacts import LrDetector
from .data_utils import Preprocessor, ArtefactsDetection
from .low_pass import LowPassFilter


class MultiStepPipeline(Preprocessor):
    """
    This class wraps other subclasses in order to detect artifacts, to remove them and to filter measurement noise, thus cleaning the EDA raw signal.
    """
    
    def __init__(self, artefacts_detection: ArtefactsDetection = LrDetector(),
                 signal_filter: Preprocessor = LowPassFilter()):
        """ Initialise the filtering pipeline.

        Parameters
        -----------
        artefacts_detection: class, optional
            class determining the algorithm to use to detect artifacts: *LrDetector()* for logistic regression from the Ideas Lab UT or *MitExplorerDetector()* for SVM from the EDA Explorer team.
        signal_filter : class, optional
            class determining the low-pass filtering method to use to further clean the EDA signal
        """

        self.artefacts_detection = artefacts_detection
        self.signal_filter = signal_filter

    def __process__(self, data: pd.Series) -> pd.Series:
        """Perform the artefact detection, artefact removal and signal filtering.

        Parameters
        ----------
        data : pd.Series
            raw EDA data , index is a list of timestamps according to the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`

        Returns
        -------
        pd.Series
            dataframe containing the EDA signal free of motion artifacts and filtered using a low-pass filter

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda
        >>> path = './empatica/1575615731_A12345'
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> eda_filtered = flirt.eda.preprocessing.MultiStepPipeline().__process__(eda['eda'])
        """

        artefact_free = self.artefacts_detection.__process__(data)
        artefact_free_filtered = self.signal_filter.__process__(artefact_free)

        return artefact_free_filtered
