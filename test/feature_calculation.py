import unittest
from datetime import datetime

import pandas as pd
from ishneholterlib import Holter

import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats
from flirt.eda.preprocessing.data_utils import Preprocessor, ArtefactsDetection
from flirt.eda.preprocessing.ekf import ExtendedKalmanFilter
from flirt.eda.preprocessing.artefacts import MitExplorerDetector, LrDetector
from flirt.eda.preprocessing.pipeline import MultiStepPipeline
#from flirt.eda.preprocessing.peak_features import ComputePeaks

class EmpaticaEdaTestCase(unittest.TestCase):
    def test_load_data(self):
        ### Load Data
        filepath = '/home/fefespinola/ETHZ_Fall_2020/flirt-2/test/wearable-data/empatica'
        eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-2/test/wearable-data/empatica/EDA.csv')
        
        ### Compute features based on different pre-processing techniques
        # Default (low-pass filter->cvx->features)
        eda_features_lpf = flirt.eda.feature_calculation.get_eda_features(data=eda)
        print('DONE', eda_features_lpf)

        # Using EDAexplorer Artifact Detection (svm->interpolation->low-pass filter->cvx->features)
        #eda_features_svm = flirt.eda.feature_calculation.get_eda_features(data=eda, window_length = 60, window_step_size = 1,  num_cores=2, preprocessor=MultiStepPipeline(MitExplorerDetector(filepath)), scr_features=ComputePeak(offset = 3))
        
        # Using Media Lab UT Artifact Detection (lr->interpolation->low-pass filter->cvx->features)
        #eda_features_lr = flirt.eda.feature_calculation.get_features(data=eda, window_length = 60, window_step_size = 1, preprocessor=MultiStepPipeline(LrDetector()), num_cores=2)
        
        # Using Extended Kalman Filter (ekf->cvx->features)
        #eda_features_ekf = flirt.eda.feature_calculation.get_features(data=eda, window_length = 60, window_step_size = 1, preprocessor=ExtendedKalmanFilter(), num_cores=2)

        # print(eda.head())

        self.assertEqual(2500, len(eda))


class EmpaticaAccTestCase(unittest.TestCase):
    def test_load_data(self):
        acc = flirt.reader.empatica.read_acc_file_into_df('wearable-data/empatica/ACC.csv')
        acc = flirt.get_acc_features(acc)

        # print(acc.head())

        self.assertEqual(313, len(acc))


class EmpaticaTSFeatureTestCase(unittest.TestCase):
    def test_load_data(self):
        ts = flirt.reader.empatica.read_acc_file_into_df('wearable-data/empatica/ACC.csv')
        ts = flirt.get_stat_features(ts)

        print(ts.head())

        self.assertEqual(313, len(ts))


class EmpaticaIbiTestCase(unittest.TestCase):
    def test_load_data(self):
        ibi = flirt.reader.empatica.read_ibi_file_into_df('wearable-data/empatica/IBI.csv')
        ibi = flirt.get_hrv_features(ibi['ibi'], 180, 1, ['td', 'fd', 'nl', 'stat'], 0.5)

        self.assertEqual(8482, len(ibi))

        self.assertEqual(8482, len(ibi))

    def test_illegal_domain(self):
        self.assertRaisesRegex(ValueError, 'invalid feature domain: foo', flirt.get_hrv_features, pd.Series(), 180, 1,
                               ['foo'], 0.5)


class HolterIbiTestCase(unittest.TestCase):
    def test_load_data(self):
        holter = Holter('wearable-data/holter/holter.ecg')
        start_time = flirt.reader.holter.get_starttime_from_holter(holter)

        self.assertEqual(datetime(2019, 8, 9, 10, 5), start_time.replace(tzinfo=None))

        ibi = flirt.reader.holter.read_holter_ibi_file_into_df(start_time, 'wearable-data/holter/holter.txt')
        # ibi = flirt.hrv.get_features(ibi['ibi'], 180, ['td', 'fd', 'stat'], 0.0)

        self.assertEqual(4766, len(ibi))
        self.assertEqual(822, ibi.iloc[0]['ibi'])
        self.assertEqual(625, ibi.iloc[-1]['ibi'])


if __name__ == '__main__':
    unittest.main()
