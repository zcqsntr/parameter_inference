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



true_params = np.array([-0.1, -0.1, -0.1, -0.1, 480000, 480000, 520000, 520000, 2, 2, 0.00049, 0.00000102115, 0.00006845928, 0.00006845928])
prior_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 100000., 100000., 100000., 100000., 1., 1., 0.005, 0.0001, 0.0001, 0.0001])
likelihood_sigmas = np.array([1., 1.])
prior_centres = np.array([-0.11, -0.09, -0.11, -0.09, 490000, 470000, 510000, 530000, 2.1, 1.9, 0.0006, 0.0000012, 0.00007, 0.00006])
# parameters to learn

likelihood_scaling = np.array([1,1,1,1,1e12, 1e12, 1e12, 1e12, 1, 1, 0.1, 0.0001, 0.0001, 0.0001])


f = open('/Users/Neythen/Desktop/masters_project/app/CBcurl_master/examples/parameter_files/double_auxotroph.yaml')
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

xSol = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux.npy')
Cins = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux_Cins.npy')
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
swarm = Swarm(domain, n_particles, n_groups, cs, ode_params)

swarm.find_minimum(initial_S, Cins, actual_N , 100,'param')


print(swarm.global_best_positions)
print(swarm.global_best_values)





#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000)
'''
for p in swarm.particles:
    im = plt.plot(*p.position, 'ro')
'''
#plt.show()
