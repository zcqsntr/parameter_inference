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
from particle_swarm_optimiser import *

class Coculture():
    def __init__(self, ode_params):
        self.ode_params = ode_params


    def sdot(self, S, t, param_vec, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
        '''
        Calculates and returns derivatives for the numerical solver odeint

        Parameters:
            S: current state
            t: current time
            Cin: array of the concentrations of the auxotrophic nutrients and the
                common carbon source
            params: list parameters for all the exquations
            num_species: the number of bacterial populations
        Returns:
            dsol: array of the derivatives for all state variables
        '''
        # extract parmeters
        '''
        A = param_vec[5]
        #A = param_vec[0]
        y = param_vec[0]
        y3 = param_vec[1]

        Rmax = param_vec[2]

        Km = self.ode_params[5]
        Km3 = self.ode_params[6]

        Km = param_vec[10:12]
        Km3 = param_vec[12:14]
        '''

        # autograd gives t as an array_box, need to convert to int
        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t = t._value
            t = int(t)
        else:
            t = int(t)
        t = min(Cin.shape[0] - 1, t) # to prevent solver from going past the max time

        Cin = Cin[t]

        print(" param vec: ", param_vec)
        A = np.reshape(param_vec[-4:], (2,2))
        y = param_vec[4:6]
        y3 = param_vec[6:8]

        Rmax = param_vec[8:10]

        Km = self.ode_params[5]
        Km3 = self.ode_params[6]

        num_species = 2
        # extract variables
        N = np.array(S[:num_species])
        C = np.array(S[num_species:2*num_species])
        C0 = np.array(S[-1])

        C0in, q = self.ode_params[:2]

        R = self.monod(C, C0, Rmax, Km, Km3)

        Cin = Cin[:num_species]
        # calculate derivatives
        dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
        dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)


        dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC)
        dsol = np.append(dsol, dC0)



        return tuple(dsol)

    def monod(self, C, C0, Rmax, Km, Km0):
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
        C = np.array(C)
        Rmax = np.array(Rmax)
        Km = np.array(Km)
        C0 = np.array(C0)
        Km0 = np.array(Km0)

        growth_rate = ((Rmax*C)/ (Km + C)) * (C0/ (Km0 + C0))

        return growth_rate

    def predict(self, params, S, Cin):
        '''
        predicts the populations at the next time point based on the current values for the params
        '''
        time_diff = 2  # frame skipping
        time_points = np.array([x *1 for x in range(time_diff)])
        sol = odeint(self.sdot, S, time_points, tuple((params, Cin)))[1:]
        pred_N = sol[-1, 0:2]

        return pred_N



true_params = np.array([-0.1, -0.1, -0.1, -0.1, 480000, 480000, 520000, 520000, 2, 2, 0.00049, 0.00000102115, 0.00006845928, 0.00006845928])
prior_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 100000., 100000., 100000., 100000., 1., 1., 0.005, 0.0001, 0.0001, 0.0001])
likelihood_sigmas = np.array([1., 1.])
prior_centres = np.array([-0.11, -0.09, -0.11, -0.09, 490000, 470000, 510000, 530000, 2.1, 1.9, 0.0006, 0.0000012, 0.00007, 0.00006])
# parameters to learn

likelihood_scaling = np.array([1,1,1,1,1e12, 1e12, 1e12, 1e12, 1, 1, 0.1, 0.0001, 0.0001, 0.0001])


f = open('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/examples/parameter_files/double_auxotroph.yaml')
param_dict = yaml.load(f)
f.close()

validate_param_dict(param_dict)
param_dict = convert_to_numpy(param_dict)

ode_params = param_dict['ode_params']

initial_X = param_dict['Q_params'][7]
initial_C = param_dict['Q_params'][8]
initial_C0 = param_dict['Q_params'][9]

initial_S = np.append(initial_X, initial_C)
initial_S = np.append(initial_S, initial_C0)

labels = ['N1', 'N2', 'C1', 'C2', 'C0']


def info(x,i,g):
    print('iter:', i)
    print('x:', x)
    print('g:', g)

xSol = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/double_aux.npy')
Cins = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/double_aux_Cins.npy')
actual_N = xSol[0:50, 0:2]

fullSol = xSol
param_vecs = []
square_losses = []
MAP_losses = []
param_losses = []
centre_losses = []

actual_params = param_dict['ode_params'][2:]

domain = np.array([[0, -0.5],[0, -0.5], [0, -0.5], [0, -0.5],[500000, 460000], [500000, 460000], [500000, 540000], [500000, 540000], [2.2, 1.8], [2.2, 1.8]])
velocity_scaling = np.array([0.1,0.1,0.1,0.1,100000,100000,100000,100000,100000,1,1])
n_particles = 50
n_groups = 5
cs = (2, 2)

system = Coculture(param_dict['ode_params'])
swarm = Swarm(system, domain, n_particles, n_groups, cs, ode_params)

swarm.find_minimum(initial_S, Cins, actual_N , 100,'param')


print(swarm.global_best_positions)
print(swarm.global_best_values)





#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000)
'''
for p in swarm.particles:
    im = plt.plot(*p.position, 'ro')
'''
#plt.show()
