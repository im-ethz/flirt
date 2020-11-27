from abc import abstractmethod

import pandas as pd


class Preprocessor(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError


class ArtefactsDetection(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError


class SignalFilter(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError


class SignalDecomposition(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> (pd.Series, pd.Series):
        raise NotImplementedError

class PeakFeatures(object):
    @abstractmethod
    def __process__(self, data: pd.Series) -> dict:
        raise NotImplementedError
<<<<<<< HEAD
    
=======
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
