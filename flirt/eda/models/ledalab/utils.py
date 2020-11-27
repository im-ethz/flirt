"""
Helper functions not necessarily related to skin conductance but useful for matlab-like python usage
"""
<<<<<<< HEAD
from __future__ import division
import subprocess
import numpy as np
from numpy import array as npa
=======
import subprocess
import numpy as np
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
import itertools
from sympy import factorint
from scipy.stats import norm
from scipy.signal import convolve


def termplot(xs, ys, legend):
    """
    Plots a line in the terminal using gnuplot, where xs and ys are the coordinates.
    legend will be the legend of the line
    example:
    import numpy as np
    x=np.linspace(0,2*np.pi,10)
    y=np.sin(x)
    utils.termplot(x,y,'sin')
    set x to None if only y should be plotted (x will be increasing from 0 to len(y))
    """
    gnuplot = subprocess.Popen("gnuplot", stdin=subprocess.PIPE)
    gnuplot.stdin.write("set term dumb\n")
    gnuplot.stdin.write("plot '-' title '" + legend + "' with lines \n")
    if xs is not None:
        for (i, j) in zip(xs, ys):
            gnuplot.stdin.write("%f %f\n" % (i, j))
    else:
        for (i, j) in enumerate(ys):
            gnuplot.stdin.write("%f %f\n" % (i, j))
    gnuplot.stdin.write("e\n")
    gnuplot.stdin.flush()
    gnuplot.stdin.close()


def get_peaks(data):
    """
    Note: ndiff can only be 1 (it was the second, unused parameter of this function)
    Original location: main/util/get_peaks.m
    """
    if not isinstance(data, np.ndarray):
<<<<<<< HEAD
        data = npa(data)
    cccrimin = npa([])
    cccrimax = npa([])
=======
        data = np.array(data)
    cccrimin = np.array([])
    cccrimax = np.array([])
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    ccd = np.diff(data)
    # Search for signum changes in first differntial:
    # slower but safer method to determine extrema than looking for zeros (taking into account
    # plateaus where ccd does not immediatly change the signum at extrema)
<<<<<<< HEAD
    cccri = npa([])
=======
    cccri = np.array([])
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    start_idx = np.flatnonzero(ccd)
    if len(start_idx) == 0:
        return (None, None)
    start_idx = start_idx[0]
    csi = np.sign(ccd[start_idx])  # currentsignum = current slope
    for i in range(start_idx + 1, ccd.size):
        if not np.sign(ccd[i]) == csi:
            if (cccri.size == 0 and csi == 1):  # if first extrema = maximum, insert minimum before
                predataidx = np.arange(start_idx, i)
                idx = np.argmin(data[predataidx])
                cccri = predataidx[idx]
            cccri = np.hstack((cccri, i))
            csi = -csi
    # if last extremum is maximum add minimum after it
    if not np.mod(cccri.size, 2):
        cccri = np.hstack((cccri, data.size - 1))
    cccri.sort()
    cccrimin = cccri[::2]  # list of minima
    cccrimax = cccri[1::2]  # list of maxima
<<<<<<< HEAD
    return (npa(cccrimin, dtype='int'), npa(cccrimax, dtype='int'))
=======
    return (np.array(cccrimin, dtype='int'), np.array(cccrimax, dtype='int'))
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781


def factor(x):
    """
    f = factor(x) returns a list containing the prime factors of x.
    """
    fd = factorint(x)
    facs = list()
    for i in fd.keys():
        nl = np.repeat(i, fd.get(i), 0)
        facs.extend(nl)
    return facs


def withinlimits(w_in, lowerlimit, upperlimit):
    """
    Replaces elements in w_in with lowerlimit or upperlimit if they are lower / higher than the specified min and max values)
    """
    return np.maximum(np.minimum(w_in, upperlimit), lowerlimit)


def powerset(x):
    """
    Return power set of set x
    """
    n = len(x)
    ps = list()
    for i in range(1, n + 1):
        s = itertools.combinations(x, i)  # singletons tuples are generated here
        retval = [val for val in s]
        ps.extend(retval)
    return ps


def divisors(a):
    """
    All divisors of a
    """
    ps = powerset(factor(a))
    d = list()
    for elem in ps:
        d.append(np.prod(elem))
    divisors = np.unique(d)
    return divisors[0:-1]


def smooth_adapt(data, winwidth_max, err_crit):
    """
    Original location: main/util/smooth_adapt.m
    """
    success = 0
    ce = np.sqrt(np.mean(pow(np.diff(data), 2)) / 2)
    iterL = np.arange(0, winwidth_max + 4, 4)
    if len(iterL) < 2:
<<<<<<< HEAD
        iterL = npa([0, 2])
=======
        iterL = np.array([0, 2])
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    count = 0
    for i in range(1, len(iterL)):
        count = i
        winwidth = iterL[i]
        scs = smooth(data, winwidth)
        scd = np.diff(scs)
        ce = np.hstack((ce, np.sqrt(np.mean(pow(scd, 2)) / 2)))  # conductance_error
        if abs(ce[i] - ce[i - 1]) < err_crit:
            success = 1
            break

    if success:  # take before-last result
        if count > 1:
            scs = smooth(data, iterL[count - 1])
            winwidth = iterL[count - 1]
        else:  # data already satisfy smoothness criteria
            scs = data
            winwidth = 0
    return (scs, winwidth)


def smooth(data, winwidth_in):
    """
    Note: only 'gauss' smoothing employed
    Original location: main/util/smooth.m
    """
    if winwidth_in < 1:
        return data
    if not isinstance(data, np.ndarray):
<<<<<<< HEAD
        data = npa(data)
    if not len(data.shape) == 1:
        raise ValueError('data is not a vector. Shape: ' + str(data.shape))
    paddata = npa(np.hstack((data[0], data, data[-1])))  # pad to remove border errors
=======
        data = np.array(data)
    if not len(data.shape) == 1:
        raise ValueError('data is not a vector. Shape: ' + str(data.shape))
    paddata = np.array(np.hstack((data[0], data, data[-1])))  # pad to remove border errors
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    winwidth = int(np.floor(winwidth_in / 2) * 2)  # force even winsize for odd window
    window = norm.pdf(np.arange(0, winwidth + 1), winwidth / 2 + 1, winwidth / 8)
    window = window / np.sum(window)  # normalize window

    data_ext = np.hstack(((np.zeros(winwidth // 2) + 1) * paddata[0],
                          paddata,
<<<<<<< HEAD
                          (np.zeros(winwidth // 2) + 1) * paddata[-1]))  # extend data to reduce convolution error at beginning and end
=======
                          (np.zeros(winwidth // 2) + 1) * paddata[-1]))
    # extend data to reduce convolution error at beginning and end
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781

    sdata_ext = convolve(data_ext, window)  # convolute with window
    sdata = sdata_ext[winwidth + 1: -winwidth - 1]  # cut to data length
    return sdata


def subrange_idx(t, t1, t2):
    """
    Original location: main/util/subrange_idx.m
    """
    t1_idx = time_idx(t, t1)
    t2_idx = time_idx(t, t2)
    idx = np.arange(t1_idx, t2_idx + 1)
    return idx


def downsamp(t, data, fac, method='step'):
    """
    method is step by default but mean can be used (by main function before starting, for example)
    Original location: main/util/downsamp.m
    """
<<<<<<< HEAD
=======

>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    def meansamp(a, fac):
        extrabit = np.mod(len(a), fac)
        if extrabit > 0:
            a = a[:-extrabit]
        a = np.mean(a.reshape((-1, fac)).T, axis=0)
        return a

    def stepsamp(a, fac):
        return a[::fac]

    if method.lower() == 'step':
        return (stepsamp(t, fac), stepsamp(data, fac))
    elif method.lower() == 'mean':
        return (meansamp(t, fac), meansamp(data, fac))
    else:
        raise ValueError('Method "' + str(method) + '" not supported')


def time_idx(time, time0):
    """
    Note: only one return value (index) as opposed to Ledlab's version which also returns the original value corresponding to the adjusted index
    Original location: main/util/time_idx.m
    """
    if not isinstance(time, np.ndarray):
<<<<<<< HEAD
        time = npa(time)
=======
        time = np.array(time)
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
    idx = np.flatnonzero(time >= time0)
    if len(idx) > 0:
        idx = np.min(idx)
        time0_adj = time[idx]
<<<<<<< HEAD
        
=======

>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
        # check if there is a closer idex before
        if not time0_adj == time[0]:
            time0_adjbefore = time[idx - 1]
            if abs(time0 - time0_adjbefore) < abs(time0 - time0_adj):
                idx = idx - 1
                time0_adj = time0_adjbefore
    else:
        idx = np.flatnonzero(time <= time0)
        idx = np.max[idx]
        time0_adj = time[idx]
    return idx


def genTime(seconds, length):
    """
    Generate a time vector represeting 'seconds' seconds with the given sample rate (srate).
    Each timepoint of the vector represents time in seconds, starting from 0
    """
    return np.linspace(0, seconds, length)


def genTimeVector(conductance, srate):
    """
    Given a conductance vector, return its corresponding time vector representing seconds at each timepoint
    """
<<<<<<< HEAD
    return genTime(len(conductance) / srate, len(conductance))
=======
    return genTime(len(conductance) / srate, len(conductance))
>>>>>>> 08d4c0758bb4dee57ba7f337632b77eec417a781
