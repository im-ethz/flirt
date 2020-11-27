import unittest
from datetime import datetime

import pandas as pd



import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats
import flirt.eda.preprocessing as eda_preprocess
<<<<<<< HEAD


class EmpaticaEdaTestCase(unittest.TestCase):
    def test_load_data(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')

        ### Compute features based on different pre-processing techniques
        # Default (low-pass filter->cvx->features)
        eda_features_lpf = flirt.eda.feature_calculation.get_eda_features(data=eda['eda'])

        # Using EDAexplorer Artifact Detection (svm->interpolation->low-pass filter->cvx->features)
        eda_features_svm = flirt.eda.feature_calculation.get_eda_features(data=eda['eda'], window_length = 60, window_step_size = 1,  num_cores=2, preprocessor=eda_preprocess.MultiStepPipeline(eda_preprocess.MitExplorerDetector(filepath)))
        
        # Using Media Lab UT Artifact Detection (lr->interpolation->low-pass filter->cvx->features)
        eda_features_lr = flirt.eda.feature_calculation.get_eda_features(data=eda['eda'], window_length = 60, window_step_size = 1, num_cores=2, preprocessor=eda_preprocess.MultiStepPipeline(eda_preprocess.LrDetector()))
        
        # Using Extended Kalman Filter (ekf->cvx->features)
        eda_features_ekf = flirt.eda.feature_calculation.get_eda_features(data=eda['eda'], window_length = 60, window_step_size = 1, num_cores=2, preprocessor=eda_preprocess.ExtendedKalmanFilter())

        # Using Ledalab decomposition algorithm (low-pass filter->ledalab->features)
        eda_features_ledalab = flirt.eda.feature_calculation.get_eda_features(data=eda['eda'], signal_decomposition=eda_preprocess.LedaLab()) 

        # print(eda.head())

        self.assertEqual(2500, len(eda))

=======
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781

class EmpaticaAccTestCase(unittest.TestCase):
    def test_load_data(self):
        acc = flirt.reader.empatica.read_acc_file_into_df('wearable-data/empatica/ACC.csv')
        acc = flirt.get_acc_features(acc)

        # print(acc.head())

        self.assertEqual(313, len(acc))


class EmpaticaIbiTestCase(unittest.TestCase):
    def test_load_data(self):
        ibi = flirt.reader.empatica.read_ibi_file_into_df('wearable-data/empatica/IBI.csv')
        ibi = flirt.get_hrv_features(ibi['ibi'], 180, 1, ['td', 'fd', 'nl', 'stat'], 0.5)

        # print(ibi.head())

        self.assertEqual(8482, len(ibi))

    def test_illegal_domain(self):
        self.assertRaisesRegex(ValueError, 'invalid feature domain: foo', flirt.get_hrv_features, pd.Series(), 180, 1,
                               ['foo'], 0.5)


class HolterIbiTestCase(unittest.TestCase):
    def test_load_data(self):
        start_time = flirt.reader.holter.get_starttime_from_holter('wearable-data/holter/holter.ecg')

        self.assertEqual(datetime(2019, 8, 9, 10, 5), start_time.replace(tzinfo=None))

        ibi = flirt.reader.holter.read_holter_ibi_file_into_df(start_time, 'wearable-data/holter/holter.txt')
        # ibi = flirt.hrv.get_features(ibi['ibi'], 180, ['td', 'fd', 'stat'], 0.0)

        self.assertEqual(4766, len(ibi))
        self.assertEqual(822, ibi.iloc[0]['ibi'])
        self.assertEqual(625, ibi.iloc[-1]['ibi'])


if __name__ == '__main__':
    unittest.main()
