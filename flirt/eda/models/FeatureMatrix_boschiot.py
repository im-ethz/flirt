# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:36:35 2017

@author: yzhang17
"""

import numpy as np
from statistics import mode
from scipy import stats
import pywt
import flirt

def statistics(data): # The function for the 4 statistics
    avg = np.mean(data) # mean
    sd = np.std(data) # standard deviation
    maxm = max(data) # maximum
    minm = min(data) # minimum
    return avg,sd,maxm,minm
    
def Derivatives(data): # Get the first and second derivatives of the data
    deriv = (data[1:-1] + data[2:])/ 2. - (data[1:-1] + data[:-2])/ 2.
    secondDeriv = data[2:] - 2*data[1:-1] + data[:-2]
    return deriv,secondDeriv

def featureMatrix(data): # Construct the feature matrix
    #length = len(labels_all)
    # Divide the data into 5 seconds time windows, 4Hz is the sampling rate, thus 20 data points each time window(5s).
    all_length = len(data)
    nb_samples = np.int(np.floor(all_length/20))
    EDA = np.array(data[0:nb_samples*20]).reshape(-1,20)
    length = len(EDA)
    assert length==nb_samples

    # Construct the feature matrix, 24 EDA features, 96 ACC features, and 120 features in total. 
    features = np.zeros((length,120))
    for i in range(length):
        deriv_EDA,secondDeriv_EDA = Derivatives(EDA[i,:])
        
        _, EDA_cD_3, EDA_cD_2, EDA_cD_1 = pywt.wavedec(EDA[i,:], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
        
        ### EDA features
        # EDA statistical features:
        features[i,0:4] = statistics(EDA[i,:])
        features[i,4:8] = statistics(deriv_EDA)
        features[i,8:12] = statistics(secondDeriv_EDA)
        # EDA wavelet features:
        features[i,12:16] = statistics(EDA_cD_3)
        features[i,16:20] = statistics(EDA_cD_2)
        features[i,20:24] = statistics(EDA_cD_1)
        
    featuresAll = stats.zscore(features) # Normalize the data using z-score
    #featuresAcc = featuresAll[:,24:120] # 96 ACC features
    featuresEda = featuresAll[:,0:24] #24 EDA features
    
    return featuresAll,featuresEda
