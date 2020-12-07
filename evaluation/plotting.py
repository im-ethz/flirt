import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, '/home/fefespinola/ETHZ_Fall_2020/flirt-1')

import flirt
import flirt.reader.empatica

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

# Set the font used for MathJax - more on this later
rc('mathtext',**{'default':'regular'})

# Set line-width parameters
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('axes', labelsize=8)


# Retrieve data
eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/project_data/WESAD/S11/EDA.csv')
filepath = '/home/fefespinola/ETHZ_Fall_2020/project_data/WESAD/S11'
fs = 4
eda_len = len(eda)
time = np.linspace(0, (eda_len)/fs, eda_len, endpoint=False)

# Init preprocessing steps
lpf = flirt.eda.preprocessing.LowPassFilter()
ekf = flirt.eda.preprocessing.ExtendedKalmanFilter()
lr_detection = flirt.eda.preprocessing.MultiStepPipeline()
svm_detection = flirt.eda.preprocessing.MultiStepPipeline(
    artefacts_detection=flirt.eda.preprocessing.MitExplorerDetector(filepath=filepath), 
    signal_filter=flirt.eda.preprocessing.LowPassFilter(cutoff = 0.01))
cvx = flirt.eda.preprocessing.CvxEda()
ledalab = flirt.eda.preprocessing.LedaLab()
peaks_mit = flirt.eda.preprocessing.ComputeMITPeaks(offset=1, start_WT=2, end_WT=4, thres=0.2)
peaks_neuro = flirt.eda.preprocessing.ComputeNeurokitPeaks(amplitude_min=0.5)

#### Process data to plot
# filter
#filtered_lpf = lpf.__process__(eda)
#filtered_ekf = ekf.__process__(eda)
filtered_lr = lr_detection.__process__(eda)
#artefacts_lr = filtered_lpf-filtered_lr
#filtered_svm = svm_detection.__process__(eda)
#artefacts_svm = filtered_lpf-filtered_svm

#print(list(artefacts_svm))


# decompose
#phasic_cvx_raw, tonic_cvx_raw = cvx.__process__(eda)
#phasic_leda_raw, tonic_leda_raw = ledalab.__process__(eda)
#cvx_raw = phasic_cvx_raw + tonic_cvx_raw #reconstructed signal
#leda_raw = phasic_leda_raw + tonic_leda_raw

phasic_cvx_lr, tonic_cvx_lr = cvx.__process__(filtered_lr)
#phasic_leda_lr, tonic_leda_lr = ledalab.__process__(filtered_lr)
#cvx_lr = phasic_cvx_lr + tonic_cvx_lr #reconstructed signal
#leda_lr = phasic_leda_lr + tonic_leda_lr

#Peak features
peaks_info_mit = peaks_mit.__process__(phasic_cvx_lr)
print('MIT', peaks_info_mit)
peaks_info_neuro = peaks_neuro.__process__(phasic_cvx_lr)
print('Neurokit', peaks_info_neuro)

print('done data generation')

### plotting
ax1 = plt.subplot(211)
#plt.plot(time, np.array(eda), 'y-', linewidth = 1, alpha = 1, label='Raw EDA')
#plt.plot(time, np.array(cvx_raw), 'g-', linewidth = 0.9, alpha = 0.8, label='Reconstructed EDA')
#plt.plot(time, np.array(filtered_lpf), 'b-', linewidth = 0.8, alpha = 0.8, label='EDA Filtered with LPF')
plt.plot(time, np.array(artefacts_lr), 'r-', linewidth = 0.8, alpha = 1, label='Artefact Locations')  
ax1.set_title('EDA Artefact Locations using Logistic Regression Method')
ax1.set_ylabel("SC Amplitude ($\mu S$)")
#ax1.set_xlim(-0, 9)
#ax1.set_ylim(-1.5, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(-0.15, 1.1, 'a)', transform=ax1.transAxes, fontsize='11', weight='bold')
plt.grid(True, linestyle='--')
plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=True, columnspacing=0.1, labelspacing=0.1,
    fancybox=True)

ax2 = plt.subplot(212)  
plt.plot(time, np.array(eda), 'y-', linewidth = 1, alpha = 1, label='Raw EDA')
#plt.plot(time, np.array(phasic_cvx_raw), 'r-', linewidth = 0.9, alpha = 1, label='Phasic Data')  
plt.plot(time, np.array(filtered_lr), 'b-', linewidth = 0.8, alpha = 0.8, label='EDA Filtered with LR')  
#plt.plot(time, np.array(leda_lr), 'r--', alpha = 0.8, label='reconstructed ledalab')
ax2.set_title('Logistic Regression Filtered EDA')
ax2.set_ylabel("SC Amplitude ($\mu S$)")
#ax2.set_xlim(-1, 5)
#ax2.set_ylim(-0.1, 5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(-0.15, 1.1, 'b)', transform=ax2.transAxes, fontsize='11', weight='bold')
plt.grid(True, linestyle='--')
plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=True, columnspacing=0.1, labelspacing=0.1,
    fancybox=True)

'''
ax3 = plt.subplot(313)  
plt.plot(time, np.array(tonic_cvx_raw), 'r-', linewidth = 0.9, alpha = 1, label='Tonic Data')  
#plt.plot(time, np.array(filtered_lr), 'b-', linewidth = 0.8, alpha = 0.7, label='filtered eda svm')  
#plt.plot(time, np.array(leda_lr), 'r--', alpha = 0.8, label='reconstructed ledalab')
ax3.set_title('Tonic Signal')
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("SCL Amplitude ($\mu S$)")
#ax3.set_xlim(-1, 5)
ax3.set_ylim(-0.1, 9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.text(-0.15, 1.1, 'c)', transform=ax3.transAxes, fontsize='11', weight='bold')
plt.grid(True, linestyle='--')
plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=True, columnspacing=0.1, labelspacing=0.1,
    fancybox=True)
'''

print('done plotting')
plt.tight_layout()
plt.savefig('lr_artefacts.pdf', dpi=300, bbox_inches='tight', transparent=False)

