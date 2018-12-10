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
from hybrid_PS_SGD import *


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''
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
large_domain = np.array([[100000, 1000000],  [0., 1.]])
domain = small_domain
x = np.linspace(domain[0][0], domain[0][1],100)
y = np.linspace(domain[1][0], domain[1][1],100)
X,Y = np.meshgrid(x,y)
Z = np.zeros(X.shape)


velocity_scaling = np.array([10000,10000]) * 0.00001
n_particles = 1
n_groups = 1
cs = (2, 2)
params = np.array(ode_params[1:3])
print(params)

time_points = np.array([t for t in range(100)])

swarm = Swarm(domain, n_particles, n_groups, cs, Cins, xSol, velocity_scaling, ode_params)
predicted_N = swarm.predict_time_series(params, xSol[0,:], Cins, time_points)
actual_N = xSol[0:50, 0]


print(swarm.MAP_loss(params, xSol[0,:],Cins, actual_N))

#plt.plot(time_points, predicted_N)
plt.figure()

#plt.plot(time_points, xSol[:,0])
#plt.show()

n_points = [i for i in range(10)]
n_points += [i for i in range(10, 200, 10)]



'''
for n_point in n_points:
    print('n_points: ', n_point)
    for i in range(Z.shape[0]):
        print(i)
        for j in range(Z.shape[1]):
            Z[i,j] = swarm.MAP_loss([X[i,j], Y[i,j]], xSol[50, :], Cins[50],xSol[50:n_point+50,0])
    np.save('Z_diff_n_points_from_stationary/Z_' + str(n_point) + '.npy', Z)

sys.exit()
'''

Z = np.load('Z_small_domain.npy')
#Z = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/global_optimsiation/Z_diff_n_points/Z_2.npy')
ind = Z.argmin()
ind = np.unravel_index(ind, (100,100))
min = np.min(Z)
'''
print(Z)
print(min)
print(X[ind], Y[ind])
'''
fig = plt.figure(figsize = (12,8))
im = plt.contour(X,Y,Z,40, vmin=abs(Z).min(), vmax=abs(Z).max())

cb = fig.colorbar(im)


swarm.find_minimum_time_series(50)


print(swarm.global_best_positions)
print(swarm.global_best_values)

positions = swarm.global_best_positions
print()
for position in positions:
    print(swarm.MAP_loss(position, xSol[0,:],Cins, actual_N))
print()



fig = plt.figure(figsize = (12,8))
plt.xlabel('gamma')
plt.ylabel('u_max')
plt.plot([480000], [0.6])
ani = animation.ArtistAnimation(fig, swarm.ims, interval=50, blit=True,repeat_delay=1000)
ani.save('hybrid_mono_TS.mp4', bitrate = 1000)


'''
results without SGD:
[array([4.82930669e+05, 5.99996282e-01]), array([4.78926146e+05, 5.99999065e-01]), array([4.83405757e+05, 5.99997100e-01]), array([4.81275060e+05, 5.99997605e-01]), array([4.80229386e+05, 5.99999829e-01])]
[5.154534762088269, 5.034893781184728, 5.194079652076269, 5.04349157265854, 5.014287648299704]

'''

# resimulate data with the best position found in terms of loss
best_index = np.argmin(swarm.global_best_values)
best_params = swarm.global_best_positions[best_index]

predicted_N = swarm.predict_time_series(best_params, xSol[0,:] , Cins, time_points)

fig = plt.figure()
plt.plot(time_points, actual_N, label = 'actual_N')
plt.plot(time_points, predicted_N, label = 'predicted_N', ls = '-.')
plt.legend()
plt.xlabel('time (hrs)')
plt.ylabel('population (10^6 cells/L)')


time_points = [t for t in range(7000)]

predicted_N = swarm.predict_time_series(best_params, xSol[0,:], Cins, time_points)
actual_N = xSol[:7000, 0]
fig = plt.figure()
plt.plot(time_points, actual_N, label = 'actual_N')
plt.plot(time_points, predicted_N, label = 'predicted_N', ls = '-.')
plt.legend()
plt.xlabel('time (hrs)')
plt.ylabel('population (10^6 cells/L)')

plt.show()
