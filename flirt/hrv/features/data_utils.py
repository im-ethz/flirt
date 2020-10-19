from abc import abstractmethod

import numpy as np


class DomainFeatures(object):
    def __get_type__(self) -> str:
        return type(self)

    @abstractmethod
    def __generate__(self, data: np.array) -> dict:
        raise NotImplementedError
