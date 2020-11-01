# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:36:35 2017

@author: yzhang17
"""

import numpy as np
from pywt import wavedec
from scipy import stats


def __statistics(data):  # The function for the 4 statistics
    avg = np.mean(data)  # mean
    sd = np.std(data)  # standard deviation
    maxm = max(data)  # maximum
    minm = min(data)  # minimum
    return avg, sd, maxm, minm


def __derivatives(data):  # Get the first and second derivatives of the data
    deriv = (data[1:-1] + data[2:]) / 2. - (data[1:-1] + data[:-2]) / 2.
    secondDeriv = data[2:] - 2 * data[1:-1] + data[:-2]
    return deriv, secondDeriv


def feature_matrix(data):  # Construct the feature matrix
    # length = len(labels_all)
    # Divide the data into 5 seconds time windows, 4Hz is the sampling rate, thus 20 data points each time window(5s).
    all_length = len(data)
    nb_samples = np.int(np.floor(all_length / 20))
    EDA = np.array(data[0:nb_samples * 20]).reshape(-1, 20)
    length = len(EDA)
    assert length == nb_samples

    # Construct the feature matrix, 24 EDA features, 96 ACC features, and 120 features in total. 
    features = np.zeros((length, 120))
    for i in range(length):
        deriv_EDA, secondDeriv_EDA = __derivatives(EDA[i, :])

        _, EDA_cD_3, EDA_cD_2, EDA_cD_1 = wavedec(EDA[i, :], 'Haar', level=3)  # 3 = 1Hz, 2 = 2Hz, 1=4Hz

        ### EDA features
        # EDA statistical features:
        features[i, 0:4] = __statistics(EDA[i, :])
        features[i, 4:8] = __statistics(deriv_EDA)
        features[i, 8:12] = __statistics(secondDeriv_EDA)
        # EDA wavelet features:
        features[i, 12:16] = __statistics(EDA_cD_3)
        features[i, 16:20] = __statistics(EDA_cD_2)
        features[i, 20:24] = __statistics(EDA_cD_1)

    featuresAll = stats.zscore(features)  # Normalize the data using z-score
    # featuresAcc = featuresAll[:,24:120] # 96 ACC features
    featuresEda = featuresAll[:, 0:24]  # 24 EDA features

    return featuresAll, featuresEda
