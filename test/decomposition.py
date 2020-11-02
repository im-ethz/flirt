import numpy as np
import pandas as pd
from numpy import savetxt
import scipy.signal as signal
from scipy.optimize import minimize, Bounds, least_squares
from numpy.linalg import norm
from numpy.random import rand, randn
from scipy.signal import cubic
import flirt
import multiprocessing
from joblib import Parallel, delayed
from tqdm.autonotebook import trange
from linvpy import linvpy
from datetime import timedelta

#from .data_utils import Preprocessor

'''
class Tonic_Phasic(Decomposition):
    """
    This class filters the raw EDA data using an Extended Kalman Filter, reducing the measurement noise and eliminating motion artifacts. 
    
    Parameters
    -----------
    data : pd.Series
        raw EDA data , index is a list of timestamps according on the sampling frequency (e.g. 4Hz for Empatica), column is the raw eda data: `eda`
    sampling_frequency : int, optional
        the frequency at which the sensor used gathers EDA data (e.g.: 4Hz for the Empatica E4)
    alpha : float, optional
        sudomotor nerve activation (S) response amplitude
    d_min : float, optional
        lowest rate of change for a probable sudomotor response
    d_max : float, optional
        the limit where all higher rates of changes are probably due to artifacts or noise
    min_diff : float, optional
        minimum absolute difference between two consecutive EDA measurements
    max_diff : float, optional
        maximum absolute difference between two consecutive EDA measurements
    min_Savi_Go : float, optional
        minimum Savitsky-Golay differential
    max_Savi_Go : float, optional
        maximum Savitsky-Golay differential

    Returns
    -------
    pd.Series
        series containing the filtered EDA signal using the Extended Kalman filter algorithm

    Notes
    ------
    - Decreasing `alpha` tends to filter out more of the noise and reduce the phasic amplitude response.
    - Increasing the gap between `min_diff` and `max_diff`, and between `min_Savi_Go` and `max_Savi_Go`, reduces the amount of artifact supression. 

    Examples
    --------
    >>> import flirt.reader.empatica
    >>> import flirt.eda
    >>> eda = flirt.reader.empatica.read_eda_file_into_df('./EDA.csv')
    >>> eda_filtered_ekf = flirt.eda.preprocessing.ExtendedKalmanFilter(sampling_frequency=4).__process__(eda['eda'])
    
    References
    ----------
    - Tronstad C, Staal OM, Saelid S, Martinsen OG. Model-based filtering for artifact and noise suppression with state estimation for electrodermal activity measurements in real time. Annu Int Conf IEEE Eng Med Biol Soc. 2015 Aug.
    
    """

    def __init__(self, sampling_frequency: int = 4):
        self.sampling_frequency = sampling_frequency

    def __process__(self, data: pd.Series) -> pd.Series:
        
        # Return SC filtered data as a pd.Series
        return pd.Series(data=SC[0:eda_len], index=data.index, name='filtered_eda')
'''
#filepath = 'wearable-data/empatica'
signal_eda = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/flirt-1/test/wearable-data/empatica/EDA.csv')
#eda = np.ravel(eda)
signal_len = len(signal_eda)
fs = 4


# Initialisation
T_y = 0.25
T_u = 2
Ds = 6 # knot size of the cubic B-spline basis function which is an indicator of the smoothness of the tonic component
'''
N = np.int(T_d/T_u)
M = eda_len
P = np.int(np.ceil(T_d/Ds) + 5)
thau_r0 = 0.1 + (1.5-0.1)*rand()
thau_d0 = 1.5 + (6-1.5)*rand()
y_p0 = eda[0]*rand()
sigma = 0.02
mu = 0.1
q0 = sigma*randn(P)+mu
u0 = np.ones((N, 1))

'''
lambda_1 = 1
lambda_2 = 1


# where thau = [thau_r, thau_d], u = [u1, u2, ..., uN], q=[q1, q2, ..., qN]

def A(k, thau_d):
    return np.exp(-k*T_y/thau_d)

def B(t, thau_r, thau_d):
    if t>=0:
        return 1/(thau_r-thau_d)*(np.exp(-t/thau_r) - np.exp(-t/thau_d))
    else:
        return 0.0
    #return h(k*T_y - 1*T_u, thau_r, thau_d) for n in range(N-1)

def C(k, p):
    return cubic(k*T_y - (p-1)*Ds)    

#B = lambda thau_r, thau_d: [[h(k*T_y - n*T_u, thau_r, thau_d) for n in range(N-1)] for k in range(eda_len-1)]
#print(B)
num_cores = 2
print('cores', num_cores)

### state vector: x=[thau_r, thau_d, y_p0, q, u] 
### thau_r = x[0], thau_d = x[1], y_p0 = x[2], q = x[3:P+4], u = x[P+4:P+N+5]


def fun_u(u, a_weighted, y_weighted, eda_len):
    # initialisation
    T_d = np.floor(eda_len/fs) # duration of recorded signal
    N = np.int(T_d/T_u)
    M = eda_len
    P = np.int(np.ceil(T_d/Ds) + 5)
    #thau_r = theta[0]
    #thau_d = theta[1]
    #y_0 = theta[2]
    #with Parallel(n_jobs=num_cores) as parallel:
    #    AAAA = parallel(delayed(A)(k, thau_d) for k in range(eda_len))
    #    BBBB = parallel(delayed(B)(k*T_y - n*T_u,  thau_r, thau_d) for k in range(eda_len) for n in range(N))
    #    CCCC = parallel(delayed(C)(k, p) for k in trange(eda_len) for p in range(P))
    return np.array(y_weighted.reshape(eda_len,) - a_weighted.reshape(M, N)@np.array(u).reshape(N,)).reshape(-1,) # 0.5*norm(y_weighted - a_weighted@np.array(u).reshape(N,))**2 + lambda_2*norm(u, 1)**1


def fun_q(q, theta, u, eda, eda_len, lambda_1, lambda_2):
    # initialisation
    T_d = np.floor(eda_len/fs) # duration of recorded signal
    N = np.int(T_d/T_u)
    M = eda_len
    P = np.int(np.ceil(T_d/Ds) + 5)
    thau_r = theta[0]
    thau_d = theta[1]
    y_0 = theta[2]
    with Parallel(n_jobs=num_cores) as parallel:
        AAAA = parallel(delayed(A)(k, thau_d) for k in range(eda_len))
        BBBB = parallel(delayed(B)(k*T_y - n*T_u,  thau_r, thau_d) for k in range(eda_len) for n in range(N))
        CCCC = parallel(delayed(C)(k, p) for k in range(eda_len) for p in range(P))
    return 0.5*norm(eda.reshape(eda_len,) - np.array(AAAA).reshape(eda_len,)*y_0 - np.array(BBBB).reshape(M, N)@np.array(u).reshape(N,) - np.array(CCCC).reshape(M, P)@np.array(q).reshape(P,))**2 + lambda_1*norm(q)**2 + lambda_2*norm(u, 1)**1

def fun_theta(theta, u, q, eda, eda_len, lambda_1, lambda_2):
    # initialisation
    T_d = np.floor(eda_len/fs) # duration of recorded signal
    N = np.int(T_d/T_u)
    M = eda_len
    P = np.int(np.ceil(T_d/Ds) + 5)
    with Parallel(n_jobs=num_cores) as parallel:
        AAAA = parallel(delayed(A)(k, theta[1]) for k in range(eda_len))
        BBBB = parallel(delayed(B)(k*T_y - n*T_u,  theta[0], theta[1]) for k in range(eda_len) for n in range(N))
        CCCC = parallel(delayed(C)(k, p) for k in range(eda_len) for p in range(P))
    return 0.5*norm(eda.reshape(eda_len,) - np.array(AAAA).reshape(eda_len,)*theta[2] - np.array(BBBB).reshape(M, N)@np.array(u).reshape(N,) - np.array(CCCC).reshape(M, P)@np.array(q).reshape(P,))**2 + lambda_1*norm(q)**2 + lambda_2*norm(u, 1)**1


def huber_psi(array, clipping):
    """
    Derivative of the Huber loss function; the "psi" version. Used in
    the weight function of the M-estimator.

    :math:`\\psi(x)=\\begin{cases}
    x& \\text{if |x|} \\leq clipping \\\\
    clipping \\cdot sign(x) & \\text{otherwise}
    \\end{cases}`

    :param array: Array of values to apply the loss function to
    :type array: numpy.ndarray
    :return: Array of same shape as the input, cell-wise results of the\
        loss function
    :rtype: numpy.ndarray

    """
    # psi version of the Huber loss function
    def unit_psi(element, clipping):
        # Casts to float to avoid comparison issues
        element = float(element)
        if abs(element) >= clipping:
            return clipping * np.sign(element)
        else:
            return element

    vfunc = np.vectorize(unit_psi)
    return vfunc(array, clipping)

# classic weights for m-estimator
def m_weights(matrix):
    # operation on a single element, which is then vectorized
    # weight(e) = psi(e) / 2e or 1 if e==0
    
    def unit_operation(element):
        clipping = 1.345
        if element == 0:
            return 1.0
        else:
            return huber_psi(element, clipping) / (2 * element)

    # unit_operation is vectorized to operate on a matrix
    vfunc = np.vectorize(unit_operation)

    # returns the matrix with the unit_operation executed on each element
    return vfunc(matrix)

# Iteratively re-weighted least squares
def irls(a, y, initial_x, max_iterations, eda_len, lambda_2, constraints_u):

    # if an initial value for x is specified, use it, otherwise generate
    #  a vector of ones
    if initial_x is not None:
        vector_x = initial_x
    else:
        # Generates a ones vector_x with length = A.columns
        vector_x = np.ones(a.shape[1])
        initial_x = np.ones(a.shape[1])

    # Ensures numpy types and flattens y into array
    a = np.matrix(a)
    y = np.matrix(y)
    y = y.reshape(-1, 1)
    initial_x = initial_x.reshape(-1, 1)

    # Residuals = y - Ax, difference between measured values and model
    residuals = y - np.dot(a, initial_x).reshape(-1, 1)
    loss = 0

    for i in range(1, max_iterations):

        # This "if" tests whether the object calling irls is a Tau- or a
        #  M-Estimator

        # If the object calling irls is not a Tau-Estimator we use the
        # normal weights

        # normalize residuals : rhat = ((y - Ax)/ self.scale)
        scale = 1.0
        rhat = np.array(residuals / scale).flatten()

        # weights_vector = weights of rhat according to the loss
        # function
        weights_vector = m_weights(rhat)

        # Makes a diagonal matrix with the values of the weights_vector
        # np.squeeze(np.asarray()) flattens the matrix into a vector
        weights_matrix = np.diag(
            np.squeeze(
                np.asarray(weights_vector)
            )
        )

        # Square root of the weights matrix, sqwm = W^1/2
        sqwm = np.sqrt(weights_matrix)

        # a_weighted = W^1/2 A
        a_weighted = np.dot(sqwm, a)

        # y_weighted = diagonal of W^1/2 y
        y_weighted = np.dot(sqwm, y)

        # vector_x_new is there to keep the previous value to compare
        vector_x_new = least_squares(fun_u, x0=vector_x, bounds=constraints_u, method='trf', loss='huber', 
             tr_options={'regularize': True}, max_nfev=5, args=(a_weighted, y_weighted, eda_len))
        #vector_x_new = minimize(fun_u, x0 = vector_x, args = (a_weighted, y_weighted, eda_len, lambda_2), method='trust-constr', bounds=constraints_u, 
         #           options={'maxiter':3})
        vector_x_new = vector_x_new.x

        # Distance between previous and current iteration
        xdis = np.linalg.norm(vector_x - vector_x_new)

        # New residuals
        residuals = y.reshape(-1) - np.dot(a, vector_x_new).reshape(-1)
        loss = loss + np.mean(residuals)

        # Divided by the specified optional self.scale, otherwise
        # self.scale = 1
        vector_x = vector_x_new

        # if the difference between iteration n and iteration n+1 is
        # smaller than self.tolerance, return vector_x
        tolerance = 1e-4
        if xdis < tolerance:
            return vector_x

    return vector_x

def __get_features_per_window(data: pd.Series, window_length: int, i: int, sampling_frequency: int = 4):

    if pd.Timedelta(data.index[i + 1] - data.index[i]).total_seconds() <= window_length:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + timedelta(seconds=window_length)

        results = {
            'datetime': max_timestamp,
        }
        
        #try:
        relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]
        data_len = len(relevant_data)
        relevant_data = np.ravel(relevant_data.values)

        # initialisation
        T_y = 0.25
        T_u = 2
        T_d = np.floor(data_len/fs) # duration of recorded signal
        Ds = 6 # knot size of the cubic B-spline basis function which is an indicator of the smoothness of the tonic component
        N = np.int(T_d/T_u)
        M = data_len
        P = np.int(np.ceil(T_d/Ds) + 5)
        thau_r0 = 0.1 + (1.5-0.1)*rand()
        thau_d0 = 1.5 + (6-1.5)*rand()
        y_p0 = relevant_data[0]*rand()
        sigma = 0.02
        mu = 0.1
        q0 = sigma*randn(P)+mu
        u0 = np.ones((N, 1))
        lambda_1 = 0
        lambda1_opt = 0
        lambda_2 = 0
        lambda2_opt = 0

        thau_r = thau_r0
        thau_d = thau_d0
        y_0 = y_p0
        theta_opt = [thau_r, thau_d, y_0]
        q_opt = q0
        u_opt = u0

        ## Creating initial conditions
        x_0 = [thau_r0, thau_d0, y_p0]
        for pp in range(P):
            x_0.append(q0[pp])
        for nn in range(N):
            x_0.append(u0[nn])

        # Create bounds
        thau_r_min = 0.1
        thau_d_min = 1.5
        thau_r_max = 1.5
        thau_d_max = 6

        # Compute C matrix as independent of optimisation variables
        with Parallel(n_jobs=num_cores) as parallel:
            CCCC = parallel(delayed(C)(k, p) for k in range(data_len) for p in range(P))
        
        # Compute Constraints and bounds
        cons_q = ({'type': 'ineq', 'fun': lambda q:  relevant_data.reshape(data_len,) - np.array(CCCC).reshape(M, P)@q})
        bnds = ((thau_r_min, thau_r_max), (thau_d_min, thau_d_max), (None, None))
        cons_u_min = []
        cons_u_max = []
        for nnn in range(N):
            cons_u_min.append(0) #cons_u + ((0, np.inf),)
            cons_u_max.append(np.inf)
        cons_u = (cons_u_min, cons_u_max)

        ### FIRST ITERATION
        for ii in range(15):
            ### Compute u optimal using inversion algorithm
            with Parallel(n_jobs=num_cores) as parallel:
                AAAA = parallel(delayed(A)(k, theta_opt[1]) for k in range(data_len))
                BBBB = parallel(delayed(B)(k*T_y - n*T_u,  theta_opt[0], theta_opt[1]) for k in range(data_len) for n in range(N))


            #M_tau = linvpy.MEstimator(loss_function=linvpy.Huber, clipping=None, regularization=linvpy.Lasso(), lamb=0, 
                        #scale=1.0, tolerance=1e-04, max_iterations=5)
            y_tau = relevant_data.reshape(data_len,)-np.array(AAAA).reshape(data_len,)*theta_opt[2]-np.array(CCCC).reshape(M, P)@np.array(q_opt).reshape(P,)
            A_tau = np.array(BBBB).reshape(M, N)
            max_iterations = 5
            u_opt = irls(A_tau, y_tau, np.array(x_0[P+3:P+3+N]).reshape(N,), max_iterations, data_len, lambda_2, constraints_u=cons_u)
            print(u_opt)

            ### Compute theta optimal using constrained minimisation
            theta_opt = minimize(fun_theta, x0 = np.array(x_0[0:3]).reshape(3,), args = (u_opt, q_opt, relevant_data, data_len, lambda_1, lambda_2), method='trust-constr', bounds=bnds, 
                    options={'maxiter':3})
            theta_opt = theta_opt.x

            ### Compute q optimal using constrained optimisation
            q_opt = minimize(fun_q, x0 = np.array(x_0[3:P+3]).reshape(P,), args = (theta_opt, u_opt, relevant_data, data_len, lambda_1, lambda_2), method='trust-constr', constraints=cons_q, 
                    options={'maxiter':3})
            q_opt = q_opt.x

            ## Update initial conditions
            x_0 = []
            for m in range(3):
                x_0.append(theta_opt[m])
            for n in range(P):
                x_0.append(q_opt[n])
            for o in range(N):
                x_0.append(u0[o])
            print('iter_1', ii)

        ##### SECOND ITERATION
        u_loss_min = np.inf
        q_loss_min = np.inf
        for jj in range(3):
            ## Update initial conditions
            x_0 = []
            for m in range(3):
                x_0.append(theta_opt[m])
            for n in range(P):
                x_0.append(q_opt[n])
            for o in range(N):
                x_0.append(u_opt[o])
            
            # Cross validation to find optimal regulariser lambda2
            for lambda_2 in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]:
                ### Compute u optimal using inversion algorithm
                with Parallel(n_jobs=num_cores) as parallel:
                    AAAA = parallel(delayed(A)(k, theta_opt[1]) for k in range(data_len))
                    BBBB = parallel(delayed(B)(k*T_y - n*T_u,  theta_opt[0], theta_opt[1]) for k in range(data_len) for n in range(N))


                #M_tau = linvpy.MEstimator(loss_function=linvpy.Huber, clipping=None, regularization=linvpy.Lasso(), lamb=lambda_2, 
                            #scale=1.0, tolerance=1e-04, max_iterations=5)
                y_tau = relevant_data.reshape(data_len,)-np.array(AAAA).reshape(data_len,)*theta_opt[2]-np.array(CCCC).reshape(M, P)@np.array(q_opt).reshape(P,)
                A_tau = np.array(BBBB).reshape(M, N)
                max_iterations = 5
                u_est = irls(A_tau, y_tau, np.array(x_0[P+3:P+3+N]).reshape(N,), max_iterations, data_len, lambda_2, constraints_u=cons_u)
                u_loss = abs(np.mean(y_tau.reshape(-1) - np.dot(A_tau, u_est).reshape(-1)))
                # compute optimal lambda and u
                if u_loss<=u_loss_min:
                    u_opt = u_est
                    u_loss_min = u_loss
                    lambda2_opt = lambda_2        
                
            print('u_opt', u_opt)
            print('loss', u_loss_min)
            print('lambda2', lambda2_opt)
            
            ### Compute theta optimal using constrained minimisation
            theta_min = minimize(fun_theta, x0 = np.array(x_0[0:3]).reshape(3,), args = (u_opt, q_opt, relevant_data, data_len, lambda1_opt, lambda2_opt), method='trust-constr', bounds=bnds, 
                    options={'maxiter':3})
            theta_opt = theta_min.x
            print('theta_opt', theta_opt)

            # Cross validation to find optimal regulariser lambda1 
            for lambda_1 in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20]:
                ### Compute q optimal using constrained optimisation
                q_min = minimize(fun_q, x0 = np.array(x_0[3:P+3]).reshape(P,), args = (theta_opt, u_opt, relevant_data, data_len, lambda_1, lambda2_opt), method='trust-constr', constraints=cons_q, 
                        options={'maxiter':3})
                q_loss = q_min.fun
                q_est = q_min.x
                # compute optimal lambda and u
                if q_loss<=q_loss_min:
                    q_opt = q_est
                    q_loss_min = q_loss
                    lambda1_opt = lambda_1

            print('q_opt', 'loss', 'lambda1')
            print(q_opt, q_loss_min, lambda1_opt)
            
            print('iter_2', jj)

        results.update({'theta':theta_opt},)
        results.update({'q':q_opt},)
        results.update({'u':u_opt},)

        #except Exception as e:
        #    pass

        return results

    else:
        return None

# parameters
window_length = 60
data_frequency = fs
window_step_size = 60

# use only every fourth value
inputs = trange(0, signal_len - 1, window_step_size * data_frequency)

# Get features
with Parallel(n_jobs=num_cores) as parallel:
    results = parallel(delayed(__get_features_per_window)(signal_eda, window_length=window_length,
                                                        i=k, sampling_frequency=data_frequency) for k in inputs)

results = pd.DataFrame(list(results))
results.set_index('datetime', inplace=True)
results.sort_index(inplace=True)
print(results)
results.to_csv('results_decomposition.csv')


optimum_results = pd.read_csv("results_decomposition.csv")
with Parallel (n_jobs=num_cores) as parallel:
    AAAA = parallel(delayed(A)(k, theta_opt[1]) for k in range(signal_len))
    BBBB = parallel(delayed(B)(k*T_y - n*T_u,  theta_opt[0], theta_opt[1]) for k in range(signal_len) for n in range(N_tot))
    CCCC = parallel(delayed(C)(k, p) for k in range(signal_len) for p in range(P))


phasic = np.array(AAAA).reshape(M_tot,)@theta_opt[3] + np.array(BBBB).reshape(M_tot, N_tot)@u_opt
tonic = np.array(CCCC).reshape(M_tot, P_tot)@q_opt
savetxt('phasic.csv', phasic, delimiter=',')
savetxt('tonic.csv', tonic, delimiter=',')

