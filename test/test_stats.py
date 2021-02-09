import unittest

import numpy as np

import flirt
import flirt.reader.empatica
import flirt.reader.holter
import flirt.stats.common


class StatsTestCase(unittest.TestCase):
    def test_get_stats(self):
        series = np.arange(1, 100)

        stats = flirt.stats.common.get_stats(series)
        # print(stats)

        self.assertEqual(50, stats['mean'])


class IbiCalcTestCase(unittest.TestCase):
    def test_ibi_calc(self):
        ibi = flirt.reader.empatica.read_ibi_file_into_df('wearable-data/empatica/IBI.csv')
        ibi = flirt.get_hrv_features(ibi['ibi'], 180, 1, ['nl'], 0.0)

        #print(ibi.head())

        self.assertEqual(8849, len(ibi))
