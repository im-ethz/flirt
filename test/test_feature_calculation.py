import unittest
from datetime import datetime

import pandas as pd


import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats
import flirt.eda.preprocessing as eda_preprocess

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
