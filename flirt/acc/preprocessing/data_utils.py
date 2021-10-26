from abc import abstractmethod

import pandas as pd


class Preprocessor(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError
