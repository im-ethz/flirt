import pickle
import os

import numpy as np
import pandas as pd

from .data_utils import ArtefactsDetection
from ..models import custom_featurematrix
from ..models.eda_explorer.EDA_Artifact_Detection_Script import eda_explorer_artifact 

class LrDetector(ArtefactsDetection):
    """ This class detects motion artifacts in the EDA raw data by classifying each 5 second window into artifact or clean, using logistic regression.
    The artifacts are removed by linear interpolation. """

    def __init__(self, sampling_frequency: int = 4):
        """ Construct the logistic regression artefact detector.
        
        Parameters
        -----------
        sampling_frequency : int, optional
            the frequency with which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        """

        self.sampling_frequency = sampling_frequency
    

    def __process__(self, data: pd.Series) -> pd.Series:
        """ Perform the artefact detection and removal.

        Parameters
        -----------
        data : pd.Series
            raw EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`
    
        Returns
        -------
        pd.Series
            dataframe containing the raw EDA signal, with motion artifacts removed using the Ideas Lab UT algorithm

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda.preprocessing
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> eda_filtered_lr = flirt.eda.preprocessing.LrDetector(sampling_frequency=4).__process__(eda['eda'])
        
        References
        -----------
        - Zhang, Y., Haghdan, M., & Xu, K. S. Unsupervised motion artifact detection in wrist-measured electrodermal activity data. In Proceedings of the 21st International Symposium on Wearable Computers. 2017.
        - https://github.com/IdeasLabUT/EDA-Artifact-Detection
        
        """

        _, eda_features = custom_featurematrix.feature_matrix(data)

        # Retrieve data from pandas dataframe
        eda = data
        eda_len = len(eda)
        datetime = eda.index
        eda = np.array(eda).reshape(eda_len)
        fs = self.sampling_frequency

        # Load model from file
        model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../models/LR_trained_model.pkl'))
        with open(model_path, 'rb') as file:
            lr_saved_model = pickle.load(file)

        predicted_probs = lr_saved_model.predict_proba(eda_features)[:, 1]
        predicted_labels = np.round(predicted_probs)

        # Interpolation in seconds
        scale = 1.0
        window = 5

        for i in range(0, len(predicted_labels) - 1):
            if (predicted_labels[i] == 1.0) and i > 0:
                start = int(i * fs * window)
                end = int(start + fs * window)
                eda[start:end] = np.nan

        # Creating dataframe
        artifact_free_eda = pd.Series(eda[:eda_len], index=datetime[:eda_len])
        artifact_free_eda = artifact_free_eda.interpolate(method='time')

        # print('- Artifaction detection and removal using logistic regression completed')

        return artifact_free_eda


class MitExplorerDetector(ArtefactsDetection):
    """ This class detects motion artifacts in the EDA raw data by classifying each 5 second window into artifact, questionable, or clean using SVM.  
    The artifacts are removed by linear interpolation. """

    def __init__(self, filepath: str, sensor: str = 'e4', sampling_frequency: int = 4):
        """ Construct the support vector machine artefact detector.

        Parameters
        -----------
        eda_filepath: str
            path to the folder containing raw EDA (file named EDA.csv), acceleration (file named ACC.csv) and temperature (file named TEMP.csv) data
        sensor: str, optional
            name of the sensor used to gather data: 'e4', 'q' or 'shimmer'
        sampling_frequency : int, optional
            the frequency with which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        """

        self.filepath = filepath
        self.sensor = sensor
        self.sampling_frequency = sampling_frequency

    def __process__(self, data: pd.Series) -> pd.Series:
        """Perform the artefact detection and removal.
        
        Parameters
        -----------
        data : pd.Series
            raw EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`

        Returns
        -------
        pd.Series
            dataframe containing the raw EDA signal, with motion artifacts removed using the EDA Explorer algorithm

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda.preprocessing
        >>> path = './empatica/1575615731_A12345'
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> eda_filtered_svm = flirt.eda.preprocessing.MitExplorerDetector(filepath=path, sensor='e4', sampling_frequency=4).__process__(eda['eda'])
        
        References
        -----------
        - Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data. In Engineering in Medicine and Biology Conference. 2015.
        - https://github.com/MITMediaLabAffectiveComputing/eda-explorer
    """
        # Load EDA data
        eda = data
        print(eda)
        eda_len = len(eda)
        datetime = eda.index
        eda = np.array(eda).reshape(eda_len)
        fs = self.sampling_frequency
        t = np.linspace(0, eda_len / fs, eda_len, endpoint=False)

        labels = eda_explorer_artifact(self.filepath, self.sensor)
        labels_multi = labels['MulticlassLabels']
        labels_binary = labels['BinaryLabels']
        print('labels', labels, len(labels), eda_len)
        # Interpolation in seconds
        scale = 1.0
        window = 5

        for i in range(0, len(labels)-1):
            if (labels_multi[i] == -1 or labels_binary[i] == -1) and i > 0:
                start = int(i * fs * window)
                end = int(start + fs * window)
                eda[start:end] = np.nan
        print('eda nan', eda)
        
        # Creating dataframe
        #artifact_free_eda = pd.Series(list(eda.values), index=datetime)
        #artifact_free_eda = artifact_free_eda.interpolate(method='linear', limit_direction='forward')
        # Creating dataframe
        artifact_free_eda = pd.Series(eda[:eda_len], index=datetime[:eda_len])
        artifact_free_eda = artifact_free_eda.interpolate(method='time')

        # print('- Artifaction detection and removal using SVM completed')

        return artifact_free_eda
