

import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from utilities import *
from inference_utils import *
import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr




'''
file to estimate just A parameters using autograd
'''

def sdot(N, t, param_vec, Cin, C, C0):

    A = np.reshape(param_vec[:4], (2,2))
    Rmax = param_vec[4:6]
    #Rmax = np.array([2,2])
    #Km = param_vec[6:8]
    Km = ode_params[5]
    Km3 = ode_params[6]

    # extract parameters
    C0in, q = ode_params[:2]

    R = monod(C, C0, Rmax, Km, Km3)

    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution

    '''
    dN1 = N[0] *(R[0] + A_vec[0]*N[0] + A_vec[1]*N[1] - q)
    dN2 = N[1] *(R[1] + A_vec[2]*N[0] + A_vec[3]*N[1] - q)
    return np.array([dN1, dN2])
    '''
    return dN


# calculates r as a function of the concentration of the rate limiting nutrient
def monod(C, C0, Rmax, Km, Km0):
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
    Rmax = np.array(Rmax) *10
    Km = np.array(Km)
    C0 = np.array(C0)
    Km0 = np.array(Km0)

    growth_rate = ((Rmax*C)/ (Km + C)) * (C0/ (Km0 + C0))

    return growth_rate




def likelihood_objective(observed_dN, param_vec):
    pass


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''
def predict(param_vec, N, Cin, C, C_0):

    # extract params from param_vec into form used by the rest of the code
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, N, time_points, tuple((param_vec,Cin, C, C_0)))[1:]

    pred_N = sol[-1, 0:2]

    return pred_N

def squared_loss(param_vec, current_S, Cin, next_N, C, C_0):
    predicted_N = predict(param_vec, current_S, Cin, C, C_0)

    return np.sum(np.array(next_N - predicted_N)**2)


def abs_loss(param_vec, current_S, Cin, next_N, C, C_0):
    predicted_N = predict(param_vec, current_S, Cin, C, C_0)
    return np.sum(np.abs(predicted_N - next_N))

grad_func = grad(squared_loss)


f = open('/Users/Neythen/Desktop/masters_project/parameter_estimation/parameter_files/double_auxotroph.yaml')
param_dict = yaml.load(f)
f.close()


validate_param_dict(param_dict)
param_dict = convert_to_numpy(param_dict)

# extract parameters
NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, \
    MAX_STEP_SIZE, MIN_EXPLORE_RATE, cutoff, _, _  = param_dict['train_params']
NOISE, error = param_dict['noise_params']
num_species, num_controlled_species, num_N_states, num_Cin_states = \
    param_dict['Q_params'][1], param_dict['Q_params'][2],  param_dict['Q_params'][3],param_dict['Q_params'][5]
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:8]
initial_X = param_dict['Q_params'][8]
initial_C = param_dict['Q_params'][9]
initial_C0 = param_dict['Q_params'][10]


# parameters to learn
param_vec = np.array([-0.2,-0.2, -0.2, -0.2, 1.5, 1.5])

labels = ['N1', 'N2', 'C1', 'C2', 'C0']



def grad_wrapper(param_vec, i):
    '''
    for the autograd optimisers
    '''

    return grad_func(param_vec, N, Cin, next_N, C, C_0)

'''
xSol, Cins = create_time_series('/Users/Neythen/masters_project/app/CBcurl_master/examples/parameter_files/unstable_2_species.yaml', '/Users/Neythen/masters_project/results/lookup_table_results/LT_unstable_repeats/repeat1/Q_table.npy', 10000)

np.save('unstable.npy', xSol)
np.save('unstable_Cins.npy', Cins)
'''

xSol = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux.npy')
Cins = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux_Cins.npy')


fullSol = xSol
losses = []
param_losses = []
actual_A = Q_params[0]
umax = ode_params[4]
actual_params = np.append(actual_A.reshape(4,), umax)

for i in range(50):
    Cin = Cins[i,:]
    sol = fullSol[i,:]
    N = sol[:2]
    C = np.array(sol[num_species:2*num_species])
    C_0 = np.array(sol[-1])

    sol1 = fullSol[i+1,:]
    next_N = sol1[:2]

    new_param_vec = adam(grad_wrapper, param_vec, num_iters = 50)
    #alpha = max(np.exp(-i/100), 0.00000001)
    alpha = 0.5

    param_vec = (1-alpha) * param_vec + alpha * new_param_vec
    '''
    grads = grad_wrapper(param_vec, 0)
    param_vec -= grads*0.00001 * np.exp(-i/1000)
    '''

    loss = squared_loss(param_vec, N, Cin, next_N, C, C_0)
    losses.append(loss)
    param_losses.append(np.sum((param_vec-actual_params)**2))


    print(alpha)
    print(loss)
    print(param_vec)

plt.plot(param_losses)
plt.title('Inferring A and u_max')
plt.xlabel('Iteration')
plt.ylabel('Sum of square loss of the parameters')
plt.savefig('parameter_loss_A_and_u.png')
plt.figure()
plt.plot(losses)
plt.title('Inferring A and u_max')
plt.xlabel('Iteration')
plt.ylabel('Square loss in predicted N')
plt.savefig('N_loss_A_and_u.png')
plt.show()


"""# load all system variables for two system trajectories
fullSol = np.load('fullSol2.npy')
fullSol2 = np.load('fullSol2.npy')
Cins = np.load('Cins.npy')
Cins2 = np.load('Cins2.npy')


# plot all system variables
plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol[:,0])), fullSol[:,i], label = labels[i])
plt.legend()

plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol2[:,0])), fullSol2[:,i], label = labels[i])
plt.legend()
"""
