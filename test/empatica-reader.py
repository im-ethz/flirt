import unittest

import flirt.simple


class EmpaticaReaderTestCase(unittest.TestCase):
    def test_empatica_reader(self):
        features = flirt.simple.get_features_for_empatica_archive('wearable-data/empatica/1560460372.zip',
                                                                  debug=True)

        self.assertEqual((30694, 177), features.shape)  # TODO: verify shape

    def test_empatica_reader_empty(self):
        features = flirt.simple.get_features_for_empatica_archive('wearable-data/empatica/1560460372.zip',
                                                                  hrv_features=False,
                                                                  eda_features=False,
                                                                  acc_features=False,
                                                                  debug=True)
        self.assertEqual((0, 0), features.shape)

    def test_empatica_reader_hrv_only(self):
        features = flirt.simple.get_features_for_empatica_archive('wearable-data/empatica/1560460372.zip',
                                                                  hrv_features=True,
                                                                  eda_features=False,
                                                                  acc_features=False,
                                                                  debug=True)
        self.assertEqual((26892, 45), features.shape)


if __name__ == '__main__':
    unittest.main()
