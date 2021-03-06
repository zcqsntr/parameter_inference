import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from matplotlib import cm
from inference_utils import *
import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.animation as animation

from hybrid_noise_test import six_hump_camel, ackleys_function
from hybrid_PS_SGD_v2 import *


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''


class Monoculture():

    def __init__(self, ode_params):
        self.ode_params = ode_params

    def sdot(self, S, t, params, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
        '''
        Calculates and returns derivatives for the numerical solver odeint

        Parameters:
            S: current state
            t: current time
            Cin: array of the concentrations of the auxotrophic nutrients and the
                common carbon source
            params: list parameters for all the equations
            num_species: the number of bacterial populations
        Returns:
            dsol: array of the derivatives for all state variables
        '''

        # extract variables


        # autograd gives t as an array_box, need to convert to int
        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t = t._value
            t = int(t)
        else:
            t = int(t)
        t = min(Cin.shape[0] - 1, t) # to prevent solver from going past the max time

        C0in = Cin[t]

        #C0in = Cin
        N = S[0]
        C0 = S[1]
        # extract parameters
        q = self.ode_params[0]
        y, Rmax = params[0:2]
        Km =  self.ode_params[3]
        R = self.monod(C0, Rmax, Km)

        # calculate derivatives
        dN = N * (R.astype(float) - q) # q term takes account of the dilution
        dC0 = q*(C0in - C0) - 1/y*R*N

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC0)

        return tuple(dsol)

    def monod(self, C0, Rmax, Km0):
        '''
        Calculates the growth rate based on the monod equation

        Parameters:
            C: the concetrations of the auxotrophic nutrients for each bacterial
                population
            C0: concentration of the common carbon source
            Rmax: array of the maximum growth rates for each bacteria
            Km: array of the saturation constants for each auxotrophic nutrient
            Km0: array of the saturation constant for the common carbon source for
                each bacterial species
        '''

        # convert to numpy
        growth_rate = Rmax * (C0/ (Km0 + C0))

        return growth_rate

    def predict(self, params, S, Cin, time_points):

        sol = odeint(sdot, S, time_points, tuple((Cin, params, 2)))[1:] #PUT THIS BACK FOR ONLINE
        return sol



def predict(params, N, Cin):
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, N, time_points, tuple((Cin, params)))[1:]

    pred_N = sol[-1, 0]

    return pred_N



def squared_loss(A, current_S, Cin, next_N):

    predicted_N = predict(A, current_S, Cin)

    return np.sum(np.array(next_N - predicted_N)**2)

def gaussian(xs, means, sigmas):

    return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-1/(2* sigmas**2)*(xs - means)**2)

def box_prior(xs, means, sigmas):

    xs_take_means = xs - means
    xs_take_means[np.where(np.abs(xs_take_means) > 1)] = 0
    return (xs_take_means/sigmas)  # centred on mean, width sigma probably want to scale this so integral is 1



true_params = np.array([480000, 0.6, 0.00049])
prior_sigmas = np.array([100000., 0.1, 0.005])
likelihood_sigmas = np.array([100000.])
prior_centres = np.array([490000,0.7, 0.0006])
# parameters to learn

likelihood_scaling = np.array([1e12,   0.1, 0.0001])


def MAP_loss(param_vec, current_S, Cin, next_N, debug = False):

    predicted_N = predict(param_vec, current_S, Cin)

    priors = gaussian(param_vec, prior_centres, prior_sigmas) # centre on true params for now

    likelihoods = gaussian(next_N, predicted_N, likelihood_sigmas)

    if debug:
        print('predicted_N', predicted_N)
        print('next_N', next_N)
        print('priors:',np.sum(np.log(priors)))
        print('likelihoods: ', np.sum(np.log(likelihoods)) )

    return  -np.sum(np.log(likelihoods))  - 0*1/weight*np.sum(np.log(priors))


# open parameter file
f = open('../parameter_files/monoculture.yaml')
param_dict = yaml.load(f)
f.close()

ode_params = param_dict['ode_params']
initial_X = param_dict['Q_params'][6]
initial_C0 = param_dict['Q_params'][7]

labels = ['N1', 'C0']


grad_func = grad(squared_loss)

def grad_wrapper(param_vec, i):
    '''
    for the autograd optimisers
    '''

    return grad_func(param_vec, sol, Cin, next_N)


def info(x,i,g):
    print('iter:', i)
    print('x:', x)
    print('g:', g)

'''
xSol, Cins = create_time_series('/Users/Neythen/masters_project/app/CBcurl_master/examples/parameter_files/random.yaml', '/Users/Neythen/masters_project/results/lookup_table_results/auxotroph_section_10_repeats/WORKING/repeat0/Q_table.npy', 10000)

np.save('smaller_target.npy', xSol)
np.save('smaller_target_Cins.npy', Cins)

'''

xSol = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/monoculture.npy')
Cins = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/monoculture_Cins.npy')


small_domain = np.array([[470000, 490000],  [0.4, 0.8]])
domain = small_domain

velocity_scaling = np.array([10000,10000]) * 0.00001
n_particles = 50
n_groups = 5
cs = (2, 2)
system = Monoculture(ode_params)
swarm = Swarm(system, domain, n_particles, n_groups, cs, velocity_scaling, ode_params)

initial_S = xSol[0,:]
Cins = np.array([Cins[1]])
actual_N = np.array([xSol[1,0]])


swarm.find_minimum(initial_S, Cins, actual_N, 20, 'param')


print(swarm.global_best_positions)
print(swarm.global_best_values)

fig = plt.figure(figsize = (12,8))
plt.plot([480000], [0.6])
ani = animation.ArtistAnimation(fig, swarm.ims, interval=50, blit=True,repeat_delay=1000)
ani.save('hybrid_mono.mp4', bitrate = 1000)
plt.show()
'''
# plot all system variables
plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol[:,0])), fullSol[:,i], label = labels[i])
plt.legend()

plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol2[:,0])), fullSol2[:,i], label = labels[i])
plt.legend()


'''
