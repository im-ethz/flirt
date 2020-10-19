import unittest

import flirt.with_


class WithPkgTestCase(unittest.TestCase):
    def test_with_empatica(self):
        self.assertRaises(ValueError, flirt.with_.empatica, 'foo')


if __name__ == '__main__':
    unittest.main()
