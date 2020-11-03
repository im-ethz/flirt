import unittest
from datetime import datetime

import pandas as pd

import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats
from flirt.eda.preprocessing.data_utils import Preprocessor, ArtefactsDetection, PeakFeatures
from flirt.eda.preprocessing.ekf import ExtendedKalmanFilter
from flirt.eda.preprocessing.artefacts import MitExplorerDetector, LrDetector
from flirt.eda.preprocessing.pipeline import MultiStepPipeline
from flirt.eda.preprocessing import ComputePeaks

class EmpaticaEdaTestCase(unittest.TestCase):
    def test_load_data(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
        self.assertEqual(10000, len(eda))

    
    def test_get_features(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
        eda_features = flirt.get_eda_features(data=eda['eda'])
        self.assertEqual(2500, len(eda_features))
    

    def test_eda_explorer(self):
        filepath = '/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica'
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
        # Using EDAexplorer Artifact Detection (svm->interpolation->low-pass filter->cvx->features)
        eda_features_svm = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1,num_cores=2,
                                                  preprocessor=MultiStepPipeline(MitExplorerDetector(filepath)), scr_features=ComputePeak(offset = 3))
        self.assertEqual(2500, len(eda_features_svm))
        print('DONE')

    def test_lr_detector(self):
        filepath = '/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica'
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
        # Using Media Lab UT Artifact Detection (lr->interpolation->low-pass filter->cvx->features)
        eda_features_lr = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1, num_cores=2,
                                                 preprocessor=MultiStepPipeline(LrDetector()))
        self.assertEqual(2500, len(eda_features_lr))

    def test_ekf(self):
        filepath = '/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica'
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
        # Using Extended Kalman Filter (ekf->cvx->features)
        eda_features_ekf = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1, num_cores=2,
                                                  preprocessor=ExtendedKalmanFilter())
        self.assertEqual(2500, len(eda_features_ekf))


if __name__ == '__main__':
    unittest.main()
