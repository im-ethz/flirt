"""
Ledalab deconvolution functions (original folder: Ledalab/analyze/deconvolution)
"""
from __future__ import division
import numpy as np
from numpy import array as npa
from scipy.signal import convolve
from scipy.signal import deconvolve
from ...models.ledalab import utils
from ...models.ledalab import analyse
from ...models.ledalab import leda2

def sdeco(nr_iv):
    leda2.settings.dist0_min = 0
    leda2.settings.segmWidth = 12

    # Downsample data for preanalysis, downsample if N > N_max but keep
    # samplingrate at 4 Hz minimum
    leda2.analysis0.target.t = npa(leda2.data.time_data)
    leda2.analysis0.target.d = npa(leda2.data.conductance_data)
    leda2.analysis0.target.sr = leda2.data.samplingrate

    Fs_min = 4
    N_max = 3000
    Fs = round(leda2.data.samplingrate)
    N = leda2.data.N

    if N > N_max:
        factorL = utils.divisors(Fs)
        FsL = Fs / factorL
        idx = np.flatnonzero(FsL >= Fs_min)
        factorL = factorL[idx]
        FsL = FsL[idx]
        if not len(factorL) == 0:
            N_new = N / factorL
            idx = np.flatnonzero(N_new < N_max)
            if not len(idx) == 0:
                idx = idx[0]
            else:
                idx = len(factorL) - 1  # if no factor meets criterium, take largest factor
            fac = factorL[idx]
            (td, scd) = utils.downsamp(npa(leda2.data.time_data), npa(leda2.data.conductance_data), fac)
            leda2.analysis0.target.t = td
            leda2.analysis0.target.d = scd
            leda2.analysis0.target.sr = FsL[idx]
        else:
            pass  # can not be downsampled any further

    leda2.analysis0.tau = npa(leda2.settings.tau0_sdeco)
    leda2.analysis0.smoothwin = leda2.settings.smoothwin_sdeco  # sec
    leda2.analysis0.tonicGridSize = leda2.settings.tonicGridSize_sdeco

    [err, x] = sdeconv_analysis(npa(leda2.analysis0.tau))  # set dist0
    optimized_tau = deconv_optimize(x, nr_iv)
    if optimized_tau is not None:
        leda2.analysis0.tau = optimized_tau
    deconv_apply()


def sdeconv_analysis(x, estim_tonic=1):
    """
    Original location: deconvolution/sdeconv_analysis.m
    """
    # Check Limits
    x[0] = utils.withinlimits(x[0], leda2.settings.tauMin, 10)
    x[1] = utils.withinlimits(x[1], leda2.settings.tauMin, 20)
    
    if x[1] < x[0]:  # tau1 < tau2
        x = x[::-1]
    if abs(x[0] - x[1]) < leda2.settings.tauMinDiff:
        x[1] = x[1] + leda2.settings.tauMinDiff

    tau = npa(x[0:2])

    data = npa(leda2.analysis0.target.d)
    t = npa(leda2.analysis0.target.t)
    sr = npa(leda2.analysis0.target.sr)

    smoothwin = leda2.analysis0.smoothwin * 8
    dt = 1 / sr
    winwidth_max = 3  # sec
    swin = round(np.minimum(smoothwin, winwidth_max) * sr)

    d = npa(data)

    # Data preparation
    tb = t - t[0] + dt

    bg = analyse.bateman_gauss(tb, 5, 1, 2, 40, .4)
    idx = np.argmax(bg)

    prefix = bg[:idx + 1] / bg[idx + 1] * d[0] #+ 10
    prefix = npa(prefix)
    prefix_idx = np.flatnonzero(prefix)
    prefix = prefix[prefix_idx]
    n_prefix = len(prefix)
    d_ext = np.hstack((prefix, d.reshape(-1)))

    t_ext = np.arange(t[0] - dt, t[0] - (n_prefix) * dt, -dt)
    t_ext = np.hstack((t_ext[::-1], t))
    tb = t_ext - t_ext[0] + dt

    kernel = analyse.bateman_gauss(tb, 0, 0, tau[0], tau[1], 0)
    # Adaptive kernel size
    midx = np.argmax(kernel)
    kernelaftermx = kernel[midx + 1:]
    
    kernel = np.hstack((kernel[:midx + 1], kernelaftermx[kernelaftermx > pow(10, -5)]))
    kernel = kernel / sum(kernel)  # normalize to sum = 1

    sigc = np.maximum(.1, leda2.settings.sigPeak / np.max(kernel) * 10)  # threshold for burst peak

    # ESTIMATE TONIC
    deconvobj = np.hstack((d_ext, d_ext[-1] * np.ones(len(kernel) - 1)))
    (driverSC, remainderSC) = deconvolve(deconvobj, kernel)
    driverSC_smooth = utils.smooth(driverSC, swin)
    # Shorten to data range
    driverSC = driverSC[n_prefix:]
    driverSC_smooth = driverSC_smooth[n_prefix:]
    remainderSC = remainderSC[np.arange(n_prefix, len(d) + n_prefix)]
    # Inter-impulse fit
    """
    Original call - changed because variables were unused:
    (onset_idx, impulse, overshoot, impMin, impMax) = analyse.segment_driver(driverSC_smooth, np.zeros(len(driverSC_smooth)), 1, sigc, round(sr * leda2.settings.segmWidth))
    """
    (impMin, impMax) = analyse.segment_driver(driverSC_smooth, np.zeros(len(driverSC_smooth)), 1, sigc, round(sr * leda2.settings.segmWidth))
    if estim_tonic:
        (tonicDriver, tonicData) = analyse.sdeco_interimpulsefit(driverSC_smooth, kernel, impMin, impMax)
    else:
        tonicDriver = npa(leda2.analysis0.target.tonicDriver)
        nKernel = len(kernel)
        convolvobj = np.hstack((tonicDriver[0] * np.ones(nKernel), tonicDriver))
        tonicData = convolve(convolvobj, kernel)
        tonicData = tonicData[nKernel:len(tonicData) + 1 - nKernel]

    # Build tonic and phasic data
    phasicData = d - tonicData
    phasicDriverRaw = driverSC - tonicDriver
    phasicDriver = utils.smooth(phasicDriverRaw, swin)

    # Compute model error
    err_MSE = analyse.fiterror(data, tonicData + phasicData, 0)
    err_RMSE = np.sqrt(err_MSE)
    err_chi2 = err_RMSE / leda2.data.conductance_error
    err1d = deverror(phasicDriver, .2)
    err1s = analyse.succnz(phasicDriver, np.maximum(.01, np.max(phasicDriver) / 20), 2, sr)
    phasicDriverNeg = npa(phasicDriver)
    phasicDriverNeg[np.flatnonzero(phasicDriverNeg > 0)] = 0
    err_discreteness = err1s
    err_negativity = np.sqrt(np.mean(pow(phasicDriverNeg, 2)))

    # CRITERION
    alpha = 5
    err = err_discreteness + err_negativity * alpha

    # SAVE VARS
    leda2.analysis0.tau = tau
    leda2.analysis0.driver = phasicDriver  # i.e. smoothed driver
    leda2.analysis0.tonicDriver = tonicDriver
    leda2.analysis0.driverSC = driverSC_smooth
    leda2.analysis0.remainder = remainderSC
    leda2.analysis0.kernel = kernel
    leda2.analysis0.phasicData = phasicData
    leda2.analysis0.tonicData = tonicData
    leda2.analysis0.phasicDriverRaw = phasicDriverRaw

    # ERROR
    leda2.analysis0.error.MSE = err_MSE
    leda2.analysis0.error.RMSE = err_RMSE
    leda2.analysis0.error.chi2 = err_chi2
    leda2.analysis0.error.deviation = np.hstack((err1d, 0))
    leda2.analysis0.error.discreteness = np.hstack((err_discreteness, 0))
    leda2.analysis0.error.negativity = err_negativity
    leda2.analysis0.error.compound = err
    return (err, x)


def deverror(v, elim):
    if not isinstance(v, np.ndarray):
        raise ValueError('v is not a numpy array')

    idx = np.logical_and(v > 0, v < elim)
    err = 1 + ( np.sum(v[idx]) / elim - np.sum(idx) ) / len(v)
    return err


def deconv_optimize(x0, nr_iv):
    """
    Only 'sdeco' method contemplated
    Original location: /analyze/deconvolution/deconv_optimize.m
    """
    if nr_iv == 0:
        return None
    xList = [x0, [1, 2], [1, 6], [1, 8], [.5, 2], [.5, 4], [.5, 6], [.5, 8]]
    x_opt = list()
    err_opt = npa([])
    for i in range(np.minimum(nr_iv, len(xList))):
        (x, error) = analyse.cgd(xList[i], sdeconv_analysis, [.3, 2], .01, 20, .05)
        x_opt.append(x)
        err_opt = np.hstack((err_opt, error[-1]))
    idx = np.argmin(err_opt)
    xopt = x_opt[idx]
    return xopt


def deconv_apply():
    """
    Original location: Ledalab/analyze/deconvolution/sdeco.m>deconv_apply
    """
    # Prepare target data for full resolution analysis

    leda2.analysis0.target.tonicDriver = npa(leda2.analysis0.target.poly.__call__(leda2.data.time_data))
    leda2.analysis0.target.t = npa(leda2.data.time_data)
    leda2.analysis0.target.d = npa(leda2.data.conductance_data)
    leda2.analysis0.target.sr = leda2.data.samplingrate

    sdeconv_analysis(leda2.analysis0.tau, 0)

    leda2.analysis = leda2.analysis0  # remember we are copying by reference, look out for issues

    # SCRs reconvolved from Driver-Peaks
    t = leda2.data.time_data
    driver = leda2.analysis.driver
    (minL, maxL) = utils.get_peaks(driver)
    minL = np.hstack((minL[:len(maxL)], len(t)))

    # Impulse data
    leda2.analysis.impulseOnset = t[minL[:-1]]
    leda2.analysis.impulsePeakTime = t[maxL]  # = effective peak-latency
    leda2.analysis.impulseAmp = driver[maxL]

    # SCR data
    leda2.analysis.onset = leda2.analysis.impulsePeakTime
    leda2.analysis.amp = np.zeros(len(maxL))
    leda2.analysis.peakTime = np.zeros(len(maxL))
    for iPeak in range(len(maxL)):
        driver_segment = leda2.analysis.driver[minL[iPeak]:minL[iPeak + 1]]  # + 1 seems like a wrong idea...
        sc_reconv = convolve(driver_segment, leda2.analysis.kernel)
        leda2.analysis.amp[iPeak] = np.max(sc_reconv)
        mx_idx = np.flatnonzero(sc_reconv == np.max(sc_reconv))
        leda2.analysis.peakTime[iPeak] = t[minL[iPeak]] + mx_idx[0] / leda2.data.samplingrate  # SCR peak could be outside of SC time range

    negamp_idx = np.flatnonzero(leda2.analysis.amp < .001)  # criterion removes peaks at end of sc_reconv due to large negative driver-segments
    leda2.analysis.impulseOnset = np.delete(leda2.analysis.impulseOnset, negamp_idx)
    leda2.analysis.impulsePeakTime = np.delete(leda2.analysis.impulsePeakTime, negamp_idx)
    leda2.analysis.impulseAmp = np.delete(leda2.analysis.impulseAmp, negamp_idx)
    leda2.analysis.onset = np.delete(leda2.analysis.onset, negamp_idx)
    leda2.analysis.amp = np.delete(leda2.analysis.amp, negamp_idx)
    leda2.analysis.peakTime = np.delete(leda2.analysis.peakTime, negamp_idx)
