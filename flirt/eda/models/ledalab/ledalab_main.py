#!/usr/bin/env python3

"""
Main ledapy module for interfacing with callers.
example usage (single process):
import ledapy
import scipy.io as sio
from numpy import array as npa
filename = 'EDA1_long_100Hz.mat'
sampling_rate = 100
matdata = sio.loadmat(filename)
rawdata = npa(matdata['data']['conductance'][0][0][0], dtype='float64')
phasicdata = ledapy.runner.getResult(rawdata, 'phasicdata', sampling_rate, downsample=4, optimisation=2)
also contains multiprocess interface pipes to be used as a separate process
"""

from __future__ import division
import numpy as np
from numpy import array as npa
import leda2
import utils
import analyse
import deconvolution
import flirt
from flirt.eda.preprocessing import data_utils, CvxEda, LowPassFilter,  MultiStepPipeline, LrDetector
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def getResult(raw_vector, result_type, sampling_rate, downsample=1, optimisation=0, pipeout=None):
    """
    Run main analysis: extract phasic driver (returned) and set all leda2 values
    parameters:
    raw_vector : raw data
    result_type : what we want returned. Either 'phasicdriver' or 'phasicdata'
    sampling_rate : input samping rate
    downsample : downsampling factor to reduce computing time (1 == no downsample)
    optimisation : level of optimization
    pipeout : if set, sends driver through pipe instead of returning it
    """
    leda2.reset()
    leda2.current.do_optimize = optimisation
    import_data(raw_vector, sampling_rate, downsample)
    deconvolution.sdeco(optimisation)
    if result_type.lower() == 'phasicdata':
        result1 = leda2.analysis.phasicData
        result2 = leda2.analysis.tonicData
    elif result_type.lower() == 'phasicdriver':
        result = leda2.analysis.driver
    else:
        raise ValueError('result_type not recognised (was ' + result_type + ')')
    if pipeout is not None:
        pipeout.send(result)
        pipeout.close()
    else:
        return result1, result2


def import_data(raw_data, srate, downsample=1):
    """
    Sets leda2 object to its appropriate values to allow analysis
    Adapted from main/import/import_data.m
    """
    if not isinstance(raw_data, np.ndarray):
        raw_data = npa(raw_data, dtype='float64')
    time_data = utils.genTimeVector(raw_data, srate)
    conductance_data = npa(raw_data, dtype='float64')
    if downsample > 1:
        (time_data, conductance_data) = utils.downsamp(time_data, conductance_data, downsample, 'mean')
    leda2.data.samplingrate = srate / downsample
    leda2.data.time_data = time_data
    leda2.data.conductance_data = conductance_data
    leda2.data.conductance_error = np.sqrt(np.mean(pow(np.diff(conductance_data), 2)) / 2)
    leda2.data.N = len(conductance_data)
    leda2.data.conductance_min = np.min(conductance_data)
    leda2.data.conductance_max = np.max(conductance_data)
    leda2.data.conductance_smoothData = conductance_data
    analyse.trough2peak_analysis()

signal_eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/project_data/WESAD/EDA.csv')
#signal_eda = np.ravel(signal_eda)
signal_len = len(signal_eda)
sampling_rate = 4

# Filter Data
preprocessor = MultiStepPipeline()
signal_decomposition = CvxEda()
filtered_dataframe = preprocessor.__process__(signal_eda)
filtered_data = np.array(filtered_dataframe)

# Decompose Data
phasic_cvx, tonic_cvx = signal_decomposition.__process__(signal_eda)

phasic_ledalab, tonic_ledalab = getResult(np.ravel(signal_eda), 'phasicdata', sampling_rate, downsample=1, optimisation=0)


# Plot Data
t = np.linspace(0, (signal_len)/sampling_rate, signal_len, endpoint=False)
ax1 = plt.subplot(311)
#plt.plot(t, np.array(signal_eda[0:signal_len]).reshape(signal_len), 'y-', linewidth=1, label='raw eda')  
plt.plot(t, phasic_ledalab, 'b-', linewidth=2, label='phasic_ledalab')  
plt.plot(t, phasic_cvx, 'r--', linewidth=1, label='phasic_cvx')
plt.grid()
plt.legend()

ax2 = plt.subplot(312)
plt.plot(t, np.array(signal_eda[0:signal_len]).reshape(signal_len), 'y-', linewidth=2, label='raw eda')  
plt.plot(t, tonic_ledalab, 'b-', linewidth=2, label='tonic_ledalab')  
plt.plot(t, tonic_cvx, 'r--', linewidth=1, label='tonic_cvx') 
plt.grid()
plt.legend()

ax3 = plt.subplot(313)
#plt.plot(t, np.array(signal_eda[0:signal_len]).reshape(signal_len), 'y-', linewidth=2, label='raw eda')  
plt.plot(t, phasic_ledalab + tonic_ledalab, 'b-', linewidth=2, label='ledalab')  
plt.plot(t, phasic_cvx + tonic_cvx, 'r--', linewidth=1, label='cvx')
plt.grid()
plt.legend()

plt.savefig('/home/fefespinola/ETHZ_Fall_2020/flirt-1/flirt/eda/models/ledalab/phasic_tonic_raw_cvx_ledalab.png')