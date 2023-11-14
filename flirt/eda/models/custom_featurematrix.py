# -*- coding: utf-8 -*-
"""
BSD 3-Clause License

Copyright (c) 2017, Yuning Zhang, Maysam Haghdan, and Kevin S. Xu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
