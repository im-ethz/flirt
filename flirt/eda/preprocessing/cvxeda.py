import cvxopt as cvx
import numpy as np
import pandas as pd

from .data_utils import SignalDecomposition
from .low_pass import LowPassFilter


class CvxEda(SignalDecomposition):
    """ This class decomposes the filtered EDA signal into tonic and phasic components using the cvxEDA algorithm. It also \
    ensures that the tonic and phasic signals never drops below zero. """

    def __init__(self, sampling_frequency: int = 4, tau0: float = 2., tau1: float = 0.7, delta_knot: float = 15.,
                 cvx_alpha: float = 8e-3, gamma: float = 1e-3, solver=None,
                 options={'reltol': 1e-9, 'show_progress': False}):
        """ Construct the signal decomposition model.

        Parameters
        -----------
        sampling_frequency : int, optional
            the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
        tau0: float, optional
            slow time constant of the Bateman function
        tau1: float, optional 
            fast time constant of the Bateman function
        delta_knot: float, optional
            time between knots of the tonic spline function
        cvx_alpha: float, optional 
            penalization for the sparse SMNA driver
        gamma: float, optional
            penalization for the tonic spline coefficients
        solver: optional
            sparse QP solver to be used.
        options: dic, optional
            solver options, see: http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
        """    

        self.sampling_frequency = sampling_frequency
        self.tau0 = tau0
        self.tau1 = tau1
        self.delta_knot = delta_knot
        self.cvx_alpha = cvx_alpha
        self.gamma = gamma
        self.solver = solver
        self.options = options

    def __process__(self, data: pd.Series) -> (pd.Series, pd.Series):
        """Decompose electrodermal activity into phasic and tonic components.

        Parameters
        -----------
        data : pd.Series
            filtered EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), \
            column is the filtered eda data: `eda`

        Returns
        -------
        pd.Series, pd.Series
            two dataframe including the phasic and the tonic components, index is a list of \
            timestamps for both dataframes

        Examples
        --------
        >>> import flirt.reader.empatica
        >>> import flirt.eda
        >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
        >>> phasic, tonic = flirt.eda.preprocessing.CvxEda().__process__(eda['eda'])
        
        References
        ----------
        - A. Greco, G. Valenza, A. Lanata, E. P. Scilingo and L. Citi. cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing. IEEE Transactions on Biomedical Engineering. April 2016.
        - https://github.com/lciti/cvxEDA
        """

        # Decompose data into tonic and phasic components
        phasic, tonic = self.__cvx_eda(data.values, 1 / self.sampling_frequency, self.tau0, self.tau1,
                                       self.delta_knot, self.cvx_alpha, self.gamma, self.solver, self.options)

        # Creating dataframe
        data_phasic = pd.Series(np.ravel(phasic), data.index)
        data_tonic = pd.Series(np.ravel(tonic), data.index)

        
        # Check if tonic values are below zero and filter if it is the case
        if np.amin(data_tonic.values) < 0:
            lowpass_filter = LowPassFilter(sampling_frequency=self.sampling_frequency, order=1, cutoff=0.2, filter='butter')
            filtered_tonic = lowpass_filter.__process__(data_tonic)
            data_tonic = filtered_tonic
        
        # Check if phasic values are below zero and filter if it is the case
        if np.amin(data_phasic.values) < 0:
            lowpass_filter = LowPassFilter(sampling_frequency=self.sampling_frequency, order=1, cutoff=0.25, filter='butter')
            filtered_phasic = lowpass_filter.__process__(data_phasic)
            data_phasic = filtered_phasic
        

        return data_phasic, data_tonic

    def __cvx_eda(self, y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2, solver=None,
                  options={'reltol': 1e-9, 'show_progress': False}):

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
