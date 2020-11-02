"""
Ledalab analysis functions (original folder: Ledalab/analyze)
"""
from __future__ import division
import numpy as np
from numpy import array as npa
import math
from scipy import stats
from scipy.signal import convolve
from scipy import interpolate
from scipy.linalg import norm
import leda2
import utils


def cgd(start_val, error_fcn, h, crit_error, crit_iter, crit_h):
    """
    Original location: analyze/cg/cgd.m
    """
    x = start_val
    (newerror, _0) = error_fcn(x)
    error = npa(newerror)
    h = npa(h)
    jiter = 0

    while True:
        olderror = newerror

        # GET GRADIENT
        if jiter == 0:
            gradient = cgd_get_gradient(x, olderror, error_fcn, h)
            direction = -gradient
            if len(gradient) == 0:
                break

        else:
            new_gradient = cgd_get_gradient(x, olderror, error_fcn, h)
            # no conjugation
            direction = -new_gradient
        if np.any(direction):
            # LINESEARCH
            [x, newerror, step] = cgd_linesearch(x, olderror, direction, error_fcn, h)
            error_diff = newerror - olderror
        else:
            error_diff = 0  # empty gradient

        error = np.hstack((error, newerror))

        if jiter > crit_iter:
            break
        if error_diff > -crit_error:  # no improvement
            h = h / 2
            if np.all(h < crit_h):
                break
        jiter += 1

    return (x, error)


def cgd_get_gradient(x, error0, error_fcn, h):
    """
    Original location: analyze/cg/cgd_get_gradient.m
    """
    Npars = len(x)
    gradient = np.zeros(Npars)

    for i in range(Npars):
        xc = npa(x)
        xc[i] = xc[i] + h[i]

        (error1, _0) = error_fcn(xc)

        if error1 < error0:
            gradient[i] = (error1 - error0)

        else:  # try opposite direction
            xc = npa(x)
            xc[i] = xc[i] - h[i]
            (error1, _0) = error_fcn(xc)

            if error1 < error0:
                gradient[i] = -(error1 - error0)

            else:
                gradient[i] = 0
    return gradient


def succnz(data, crit, fac, sr):
    """
    Original location: analyze/deconvolution/succnz.m
    """
    n = len(data)

    abovecrit = np.abs(data) > crit
    nzidx = np.flatnonzero(np.diff(abovecrit)) + 1

    if len(nzidx) == 0:
        return 0

    if abovecrit[0] == 1:
        nzidx = np.hstack((1, nzidx))

    if abovecrit[-1] == 1:
        nzidx = np.hstack((nzidx, n+1))

    nzL = nzidx[np.arange(1, len(nzidx), 2)] - nzidx[np.arange(0, len(nzidx), 2)]

    return np.sum(pow(nzL / sr, fac) / (n / sr))

def cgd_linesearch(x, error0, direction, error_fcn, h):
    """
    Original location: analyze/cg/cgd_linesearch
    """
    direction_n = direction / norm(direction, 2)
    error_list = npa(error0)
    stepsize = h
    maxSteps = 6
    count = 0
    factor = npa([0])
    for iStep in range(1, maxSteps):

        factor = np.hstack((factor, pow(2, (iStep - 1))))
        xc = x + (direction_n * stepsize) * factor[iStep]
        (catVal, xc) = error_fcn(xc)  # xc may be changed due to limits
        error_list = np.hstack((error_list, catVal))

        if error_list[-1] >= error_list[-2]:  # end of decline
            if iStep == 1:  # no success
                step = 0
                error1 = npa(error0)
                
            else:  # parabolic
                p = np.polyfit(factor, error_list, 2)
                fx = np.arange(factor[0], factor[-1] + .1, .1)
                fy = np.polyval(p, fx)
                idx = np.argmin(fy)
                fxm = fx[idx]
                xcm = x + (direction_n * stepsize) * fxm
                (error1, xcm) = error_fcn(xcm)  # xc may be changed due to limits

                if error1 < error_list[iStep - 1]:
                    xc = xcm
                    step = fxm
                    
                else:  # finding Minimum did not work
                    xc = x + (direction_n * stepsize) * factor[iStep - 1]  # before last point
                    (error1, xc) = error_fcn(xc)  # recalculate error in order to check for limits again
                    step = factor[iStep - 1]

            return (xc, error1, step)
        count = iStep
    step = factor[count]
    error1 = error_list[count]

    return (xc, error1, step)


def segment_driver(data, remd, ndiff, sigc, segmWidth):
    """
    Orignal location: analyze/deconvolution/segment_driver.m
    """
    """
    SKIPPING SINCE UNUSED
    segmOnset = npa([])
    segmImpulse = list()
    segmOversh = list()
    impMin = npa([])
    impMax = npa([])
    """

    (cccrimin, cccrimax) = utils.get_peaks(data)
    if len(cccrimax) == 0:
        return None

    (minL, maxL) = signpeak(data, cccrimin, cccrimax, sigc)
    """
    SKIPPING SINCE UNUSED - BEWARE THE BUGS
    
    (rmdimin, rmdimax) = utils.get_peaks(remd)  # get peaks of remainder
    (rmdimins, rmdimaxs) = deconvolution.signpeak(remd, rmdimin, rmdimax, .005)  # get remainder segments
    # Segments: 12 sec, max 3 sec preceding maximum
    for i in range(len(maxL)):
        segm_start = np.maximum(minL[i, 0], maxL[i] - round(segmWidth / 2))
        segm_end = np.minimum(segm_start + segmWidth - 1, len(data))
        # impulse
        segm_idx = range(int(segm_start), int(segm_end))
        segm_data = npa(data[segm_idx])
        segm_idx_above = np.nonzero(segm_idx >= minL[i, 1])
        segm_data[segm_idx_above] = 0
        segmOnset = np.hstack((segmOnset, segm_start))
        segmImpulse.append(segm_data)
        # overshoot
        oversh_data = np.zeros(len(segm_idx))
        if i < len(maxL):
            rmi = np.flatnonzero(np.logical_and(rmdimaxs > maxL[i], rmdimaxs < maxL[i + 1]))
        else:
            rmi = np.flatnonzero(rmdimaxs > maxL[i])
        # no zero overshoots
        if rmi.size == 0:
            if i < len(maxL):
                rmi = np.flatnonzero(np.logical_and(rmdimax > maxL[i], rmdimax < maxL[i + 1]))
            else:
                rmi = np.flatnonzero(rmdimax > maxL[i])
            rmdimaxs = npa(rmdimax)
            if rmdimin is not None:
                rmdimins = np.hstack((rmdimin[:-1], rmdimin[1:]))
        if rmi.size > 0:
            rmi = rmi[0]
            oversh_start = np.maximum(rmdimins[rmi, 0], segm_start)
            oversh_end = np.minimum(rmdimins[rmi, 1], segm_end)  # min(rmdimins(rmi+1), segm_end);
            oversh_idx = np.arange((oversh_start - segm_start + 1), len(oversh_data) - (segm_end - oversh_end) + 1)
            oversh_data[oversh_idx] = remd[oversh_start:oversh_end + 1]
        segmOversh.append(oversh_data)
    #return (segmOnset, segmImpulse, segmOversh, impMin, impMax)
    """
    return (minL, maxL)


def fiterror(data, fit, npar):
    """
    npars = number of unfree parameters with df = n - npar
    'MSE' method only supported
    data and fit must be numpy arrays (row vectors)
    """
    if not isinstance(data, np.ndarray):
        raise ValueError('data is not a numpy array')
    if not isinstance(fit, np.ndarray):
        raise ValueError('fit is not a numpy array')

    residual = data - fit
    n = len(data)
    SSE = np.sum(pow(residual, 2))
    return SSE / n


def bateman(time, onset, amp, tau1, tau2):
    """
    original location: analyze/template/bateman.m
    time must be a numpy array
    """
    if not isinstance(time, np.ndarray):
        time = npa(time)
    if tau1 < 0 or tau2 < 0:
        raise ValueError('tau1 or tau2 < 0: ({:f}, {:f})'.format(tau1, tau2))

    if tau1 == tau2:
        raise ValueError('tau1 == tau2 == {:f}'.format(tau1))

    conductance = np.zeros(time.shape)
    rang = np.flatnonzero(time > onset)
    if len(rang) == 0:
        return None
    xr = time[rang] - onset

    if amp > 0:
        maxx = tau1 * tau2 * math.log(tau1 / tau2) / (tau1 - tau2)  # b' = 0
        maxamp = abs(math.exp(-maxx / tau2) - math.exp(-maxx / tau1))
        c = amp / maxamp
    else:  # amp == 0: normalized bateman, area(bateman) = 1/sr
        sr = round(1 / np.mean(np.diff(time)))
        c = 1 / ((tau2 - tau1) * sr)

    if tau1 > 0:
        conductance[rang] = c * (np.exp(-xr / tau2) - np.exp(-xr / tau1))
    else:
        conductance[rang] = c * np.exp(-xr / tau2)

    return conductance


def bateman_gauss(time, onset, amp, tau1, tau2, sigma):
    """
    Original location: analyze/template/bateman_gauss.m
    """

    component = bateman(time, onset, 0, tau1, tau2)

    if sigma > 0:
        sr = round(1 / np.mean(np.diff(time)))
        winwidth2 = int(np.ceil(sr * sigma * 4))  # round half winwidth: 4 SD to each side
        t = np.arange(1, (winwidth2 * 2 + 2))  # odd number (2*winwidth-half+1) + 2 because of numpy's arange implementation
        g = stats.norm.pdf(t, winwidth2 + 1, sigma * sr)
        g = g / np.max(g) * amp
        bg = convolve(np.hstack(((np.zeros(winwidth2) + 1) * component[0], component, (np.zeros(winwidth2) + 1) * component[-1])), g)
        
        component = bg[(winwidth2 * 2): -winwidth2 * 2]
    
    return component


def trough2peak_analysis():
    ds = leda2.data.conductance_smoothData
    t = leda2.data.time_data
    (minL, maxL) = utils.get_peaks(ds)
    minL = minL[:len(maxL)]
    leda2.trough2peakAnalysis.onset = t[minL]
    leda2.trough2peakAnalysis.peaktime = t[maxL]
    leda2.trough2peakAnalysis.onset_idx = minL
    leda2.trough2peakAnalysis.peaktime_idx = maxL
    leda2.trough2peakAnalysis.amp = ds[maxL] - ds[minL]


def sdeco_interimpulsefit(driver, kernel, minL, maxL):
    """
    Original location > analyze>decomposition>sdeco_interimpulsefit.m
    all parameters must be numpy arrays
    """
    t = leda2.analysis0.target.t
    d = leda2.analysis0.target.d
    sr = leda2.analysis0.target.sr
    tonicGridSize = leda2.settings.tonicGridSize_sdeco
    nKernel = len(kernel)

    # Get inter-impulse data index
    iif_idx = npa([])
    if len(maxL) > 2:
        for i in range(len(maxL) - 1):
            gap_idx = np.arange(minL[i, 1], minL[i + 1, 0])
            if gap_idx.size == 0:
                gap_idx = minL[i, 1]
            iif_idx = np.hstack((iif_idx, gap_idx))
        iif_idx = np.hstack((minL[1, 0], iif_idx, np.arange(minL[-1, 1], len(driver) - sr)))
    else:  # no peaks (exept for pre-peak and may last peak) so data represents tonic only, so ise all data for tonic estimation
        iif_idx = np.flatnonzero(t > 0)

    iif_idx = [x for x in iif_idx.astype(int)]
    iif_t = t[iif_idx]
    iif_data = driver[iif_idx]

    groundtime = np.hstack((np.arange(0, t[-2], tonicGridSize), t[-1]))

    if tonicGridSize < 30:
        tonicGridSize = tonicGridSize * 2

    groundlevel = npa([])
    for i in range(len(groundtime)):
        # Select relevant interimpulse time points for tonic estimate at groundtime
        if i == 0:
            t_idx = np.logical_and(iif_t <= groundtime[i] + tonicGridSize, iif_t > 1)
            grid_idx = np.logical_and(t <= groundtime[i] + tonicGridSize, t > 1)
        elif i + 1 == len(groundtime):
            t_idx = np.logical_and(iif_t > groundtime[i] - tonicGridSize, iif_t < t[-1] - 1)
            grid_idx = np.logical_and(t > groundtime[i] - tonicGridSize, t < t[-1] - 1)
        else:
            t_idx = np.logical_and(iif_t > groundtime[i] - tonicGridSize / 2, iif_t <= groundtime[i] + tonicGridSize / 2)
            grid_idx = np.logical_and(t > groundtime[i] - tonicGridSize / 2, t <= groundtime[i] + tonicGridSize / 2)
        # Estimate groundlevel at groundtime
        if len(np.flatnonzero(t_idx)) > 2:
            groundlevel = np.hstack((groundlevel, (np.minimum(np.mean(iif_data[t_idx]), d[utils.time_idx(t, groundtime[i])]))))
        else:  # if no inter-impulses data is available ...
            groundlevel = np.hstack((groundlevel, (np.minimum(np.median(driver[grid_idx]), d[utils.time_idx(t, groundtime[i])]))))
    
    # matlab: d = pchip(x,y,t)
    # python:
    # p = interpolate.PchipInterpolator(x,y)
    # d = p.__call__(t)
    np.seterr(divide='ignore')
    tonicDriver = interpolate.pchip_interpolate(groundtime, groundlevel, t)
    groundlevel_pre = groundlevel.copy()

    tonicData = convolve(np.hstack((tonicDriver[0] * np.ones(nKernel), tonicDriver)), kernel)
    tonicData = tonicData[np.arange(nKernel, len(tonicData) - nKernel + 1)]

    # Correction for tonic sections still higher than raw data
    # Move closest groundtime at time of maximum difference of tonic surpassing data
    for i in range(len(groundtime) - 2, -1, -1):

        t_idx = utils.subrange_idx(t, groundtime[i], groundtime[i + 1])
        tonicSub = (tonicData[t_idx] + leda2.settings.dist0_min) - d[t_idx]
        ddd = tonicSub.max()

        if ddd > np.spacing(1):
            groundlevel[i] = groundlevel[i] - ddd
            groundlevel[i + 1] = groundlevel[i + 1] - ddd

            tonicDriver = interpolate.pchip_interpolate(groundtime, groundlevel, t)
            tonicData = convolve(np.hstack((tonicDriver[0] * np.ones(nKernel), tonicDriver)), kernel)
            tonicData = tonicData[np.arange(nKernel, len(tonicData) + 1 - nKernel)]

    # Save to vars
    leda2.analysis0.target.poly = interpolate.PchipInterpolator(groundtime, groundlevel)
    leda2.analysis0.target.groundtime = groundtime
    leda2.analysis0.target.groundlevel = groundlevel
    leda2.analysis0.target.groundlevel_pre = groundlevel_pre
    leda2.analysis0.target.iif_t = iif_t
    leda2.analysis0.target.iif_data = iif_data

    return (tonicDriver, tonicData)


def signpeak(data, cccrimin, cccrimax, sigc):
    """
    Originally subfunction of segment_driver
    data, cccrimin, cccrimax must be row vectors of shape (x,)
    """
    if cccrimax is None:
        return (None, None)
    if not isinstance(data, np.ndarray):
        raise ValueError('data is not a numpy array')
    if not isinstance(cccrimin, np.ndarray):
        raise ValueError('cccrimin is not a numpy array')
    if not isinstance(cccrimax, np.ndarray):
        raise ValueError('cccrimax is not a numpy array')

    dmm = np.vstack((data[cccrimax] - data[cccrimin[:-1]],
                     data[cccrimax] - data[cccrimin[1:]]))
    maxL = npa(cccrimax[np.max(dmm, axis=0) > sigc])

    # keep only minima right before and after sign maxima
    minL = None
    for i in range(len(maxL)):
        minm1_idx = np.flatnonzero(cccrimin < maxL[i])
        before_smpl = cccrimin[minm1_idx[-1]]
        after_smpl = cccrimin[minm1_idx[-1] + 1]
        newStack = np.hstack((before_smpl, after_smpl))
        if minL is None:
            minL = newStack
        else:
            minL = np.vstack((minL, newStack))
    return (minL, maxL)