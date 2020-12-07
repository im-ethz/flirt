import unittest

import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats
import flirt.eda.preprocessing as eda_preprocess


class EmpaticaEdaTestCase(unittest.TestCase):
    def test_load_data(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        self.assertEqual(10000, len(eda))

    
    def test_get_features(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        eda_features = flirt.get_eda_features(data=eda['eda'])
        self.assertEqual(2500, len(eda_features))
    

    def test_eda_explorer(self):
        filepath = 'wearable-data/empatica'
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        # Using EDAexplorer Artifact Detection (svm->interpolation->low-pass filter->cvx->features)
        eda_features_svm = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1, num_cores=2,
                                                  preprocessor=eda_preprocess.MultiStepPipeline(
                                                      eda_preprocess.MitExplorerDetector(filepath)))
        self.assertEqual(2500, len(eda_features_svm))

    
    def test_lr_detector(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        # Using Media Lab UT Artifact Detection (lr->interpolation->low-pass filter->cvx->features)
        eda_features_lr = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1, num_cores=2,
                                                 preprocessor=eda_preprocess.MultiStepPipeline(
                                                     eda_preprocess.LrDetector()))
        self.assertEqual(2500, len(eda_features_lr))

    
    def test_ekf(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        # Using Extended Kalman Filter (ekf->cvx->features)
        eda_features_ekf = flirt.get_eda_features(data=eda['eda'], window_length=60, window_step_size=1, num_cores=2,
                                                  preprocessor=eda_preprocess.ExtendedKalmanFilter())
        self.assertEqual(2500, len(eda_features_ekf))
    
    def test_ledalab(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        # Using Ledalab as decomposition method (low-pass filter->ledalab->features)
        eda_features_leda = flirt.get_eda_features(data=eda['eda'], 
                                                signal_decomposition=eda_preprocess.LedaLab())
        self.assertEqual(2500, len(eda_features_leda))

    def test_ledalab(self):
        eda = flirt.reader.empatica.read_eda_file_into_df('wearable-data/empatica/EDA.csv')
        # Using Ledalab as decomposition method (low-pass filter->ledalab->features)
        eda_features_leda = flirt.get_eda_features(data=eda['eda'],
                                                   signal_decomposition=eda_preprocess.LedaLab())
        self.assertEqual(2500, len(eda_features_leda))


if __name__ == '__main__':
    unittest.main()
