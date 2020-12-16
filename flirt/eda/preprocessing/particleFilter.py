import numpy as np
import pyparticleest.utils.kalman as kalman
import pyparticleest.interfaces as interfaces
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyparticleest.simulator as simulator

import sys
sys.path.insert(1, '/home/fefespinola/ETHZ_Fall_2020/flirt-1')

import flirt
import flirt.reader.empatica


class Model(interfaces.ParticleFiltering):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, P0, Q, R):
        self.P0 = np.copy(P0)
        self.Q = np.copy(Q)
        self.R = np.copy(R)

    def create_initial_estimate(self, N):
        return np.random.normal(0.0, self.P0, (N,)).reshape((-1, 1))

    def sample_process_noise(self, particles, u, t):
        """ Return process noise for input u """
        N = len(particles)
        return np.random.normal(0.0, self.Q, (N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise):
        """ Update estimate using 'data' as input """
        particles += noise

    def measure(self, particles, y, t):
        """ Return the log-pdf value of the measurement """
        logyprob = np.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf_scalar(particles[k].reshape(-1, 1) - y, self.R)
        return logyprob

    def logp_xnext_full(self, part, past_trajs, pind,
                        future_trajs, find, ut, yt, tt, cur_ind):

        diff = future_trajs[0].pa.part[find] - part

        logpxnext = np.empty(len(diff), dtype=float)
        for k in range(len(logpxnext)):
            logpxnext[k] = kalman.lognormpdf(diff[k].reshape(-1, 1), np.asarray(self.Q).reshape(1, 1))
        return logpxnext


def particle_filter_noise_reduction():
    #x = load_pickle_signal_data(500)

    #data = get_data()
    #x_df = data['hrv_mean_ibi'][3000:3200]
    #x = x_df.values

    #steps = len(x)-1

    num_paricles = 80
    num_smoothers = 40
    P0 = 0.02 # model variance
    Q = 0.02 # process noise covariance
    R = np.asarray(((0.06,),)) # measurement noise covariance

    #y = generate_dataset(x, R)
    y = flirt.reader.empatica.read_eda_file_into_df('/home/fefespinola/ETHZ_Fall_2020/project_data/WESAD/S11/EDA.csv')
    y = np.array(y.values)
    y = y[10000:15000]
    model = Model(P0, Q, R)
    sim = simulator.Simulator(model, u=None, y=y)

    sim.simulate(num_paricles, num_smoothers, filter='PF', smoother='full', meas_first = False)

    (vals, _) = sim.get_filtered_estimates()
    vals_mean = sim.get_filtered_mean()
    #svals = sim.get_smoothed_estimates()
    #svals_mean = sim.get_smoothed_mean()

    fs = 4
    eda_len = len(y)
    time = np.linspace(0, (eda_len)/fs, eda_len, endpoint=False)

    caption = r'PPG sample signal taken from the fieldstudy dataset'
    plt.plot(time, y, 'y-', label='measurement: ' + r'$y_{t}$')
    #plt.plot(time, vals[:-1, :, 0], 'k.', markersize=0.4)
    #plt.plot(time, svals[:-1, 0, 0], 'g-', linewidth = 0.8, alpha = 1, label='state predictction: '+ r'$\hat{x}_t$')
    #plt.plot(time, svals_mean[:-1, 0], 'y-', linewidth = 0.8, alpha = 1, label='state mean predictction: '+ r'$\hat{x}_t$')
    plt.plot(time, vals_mean[:-1, 0], 'b-',  linewidth = 0.8, alpha = 1, label='state mean predictction: '+ r'$\hat{x}_t$')
    plt.xlabel('t')
    plt.ylabel('eda')
    plt.legend()
    plt.title('Particle Filter: EDA Signal')
    plt.grid()
    plt.savefig('particle_filter.png', dpi=300, bbox_inches='tight', transparent=False)


def get_scalogram():

    testData = load_pickle_signal_data()
    data = testData[:100]
    scale = np.arange(1, 100, 1)
    coef, freq = pywt.cwt(data, scale, 'mexh')

    plt.imshow(coef)



if __name__ =="__main__":
    particle_filter_noise_reduction()