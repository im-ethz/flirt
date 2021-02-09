import multiprocessing
import warnings
from datetime import timedelta

import cvxopt as cvx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.autonotebook import trange
from ..util import processing

from ..stats.common import get_stats


def get_eda_features(data: pd.Series, window_length: int = 60, window_step_size: int = 1, data_frequency: int = 4,
                     num_cores: int = 0):
    """
    Computes statistical EDA features based on the signal decompositon into phasic/tonic components of the skin \
    conductivity.

    Parameters
    ----------
    data : pd.Series
        input EDA time series
    window_length : int
        the window size in seconds to consider
    window_step_size : int
        the time step to shift each window
    data_frequency : int
        the frequency of the input signal
    num_cores : int, optional
        number of cores to use for parallel processing, by default use all available

    Returns
    -------
    EDA Features: pd.DataFrame
        A DataFrame containing statistical aggregation features of the tonic/phasic EDA components.

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> eda = flirt.reader.empatica.read_eda_file_into_df("EDA.csv")
    >>> acc_features = flirt.get_eda_features(eda['eda'], 180, 1)
    """

    input_data = data.copy()

    # ensure we have a DatetimeIndex, needed for calculation
    if not isinstance(input_data.index, pd.DatetimeIndex):
        input_data.index = pd.DatetimeIndex(input_data.index)

    # use only every nth value
    inputs = trange(0, len(input_data) - 1, window_step_size * data_frequency, desc="EDA features")

    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()

    def process(memmap_data) -> dict:
        with Parallel(n_jobs=num_cores, max_nbytes=None) as parallel:
            return parallel(delayed(__get_scr_scl)
                            (memmap_data, window_length=window_length, data_frequency=data_frequency, i=k) for k in
                            inputs)

    results = processing.memmap_auto(input_data, process)

    results = pd.DataFrame(list(filter(None, results)))
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    for column in results.columns:
        nan_count = results[column].isin([np.nan, np.inf, -np.inf]).sum()
        proportion = nan_count / len(results)
        if proportion > 0.05:
            warnings.warn(str(column) + " contains more than 5% (actual: " + str((proportion * 100).round(2))
                          + "%) nan, inf, or -inf values. We recommend to delete this feature column.", stacklevel=3)
    return results


def __get_scr_scl(data: pd.Series, window_length: int, data_frequency: int, i: int):
    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= window_length:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)

        results = {
            'datetime': max_timestamp,
        }

        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

        try:
            r, t = __cvx_eda(relevant_data.values, 1 / data_frequency)
            results.update(get_stats(np.ravel(t), 'tonic'))
            results.update(get_stats(np.ravel(r), 'phasic'))
        except Exception:
            pass

        return results

    else:
        return None


def __cvx_eda(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2, solver=None,
              options={'reltol': 1e-9, 'show_progress': False}):
    """
    CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see: http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """

    n = len(y)
    y = cvx.matrix(y)

    # bateman ARMA model
    a1 = 1. / min(tau1, tau0)  # a1 > a0
    a0 = 1. / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta ** 2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta ** 2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cvx.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))
    M = cvx.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0., -1.)]  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cvx.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cvx.matrix(np.c_[np.ones(n), np.arange(1., n + 1.) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    # old_options = cvx.solvers.options.copy()
    cvx.solvers.options.clear()
    cvx.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m, n: cvx.spmatrix([], [], [], (m, n))
        G = cvx.sparse([[-A, z(2, n), M, z(nB + 2, n)], [z(n + 2, nC), C, z(nB + 2, nC)],
                        [z(n, 1), -1, 1, z(n + nB + 2, 1)], [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
                        [z(n + 2, nB), B, z(2, nB), cvx.spmatrix(1.0, range(nB), range(nB))]])
        h = cvx.matrix([z(n, 1), .5, .5, y, .5, .5, z(nB, 1)])
        c = cvx.matrix([(cvx.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cvx.solvers.conelp(c, G, h, dims={'l': n, 'q': [n + 2, nB + 2], 's': []})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cvx.sparse([[Mt * M, Ct * M, Bt * M], [Mt * C, Ct * C, Bt * C],
                        [Mt * B, Ct * B, Bt * B + gamma * cvx.spmatrix(1.0, range(nB), range(nB))]])
        f = cvx.matrix([(cvx.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cvx.solvers.qp(H, f, cvx.spmatrix(-A.V, A.I, A.J, (n, len(f))),
                             cvx.matrix(0., (n, 1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    # cvx.solvers.options.clear()
    # cvx.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n + nC]
    t = B * l + C * d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return r, t
    # return r, p, t, l, d, e, obj
