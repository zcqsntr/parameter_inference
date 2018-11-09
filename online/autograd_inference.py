

import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *
import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr



'''
make new versions of the functions being differentiated removing features not
supported by autograd
'''
def sdot(S, t, learned_params, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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
    A = np.reshape(param_vec[0:4],(2, 2))

    y = learned_params[4:6]
    y3 = learned_params[6:8]
    Rmax = learned_params[8:10]
    Km = learned_params[10:12]
    Km3 = learned_params[12:14]

    ode_params[2:] = [y,y3,Rmax, Km, Km3]
    num_species = 2
    # extract variables
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    params = [1., 0.5]

    # extract parameters
    C0in, q = params


    R = monod(C, C0, Rmax, Km, Km3)

    Cin = Cin[:num_species]


    # calculate derivatives
    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)

    # consstruct derivative vector for odeint
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC)
    dsol = np.append(dsol, dC0)
    return tuple(dsol)

def sdot_seperate(S, t, learned_params, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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
    A = np.reshape(param_vec[0:4],(2, 2))

    y = learned_params[4:6]
    y3 = learned_params[6:8]
    Rmax = learned_params[8:10]
    Km = learned_params[10:12]
    Km3 = learned_params[12:14]

    ode_params[2:] = [y,y3,Rmax, Km, Km3]
    num_species = 2
    # extract variables
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    params = [1., 0.5]

    # extract parameters
    C0in, q = params


    R = monod(C, C0, Rmax, Km, Km3)

    Cin = Cin[:num_species]


    # calculate derivatives
    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)

    # consstruct derivative vector for odeint
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC)
    dsol = np.append(dsol, dC0)
    return tuple(dsol)

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
    Rmax = np.array(Rmax)
    Km = np.array(Km)
    C0 = np.array(C0)
    Km0 = np.array(Km0)
    growth_rate = (Rmax*C)/ (Km + C)

    return growth_rate




def likelihood_objective(observed_dN, param_vec):
    pass


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''
def predict(param_vec, S, Cin):

    # extract params from param_vec into form used by the rest of the code
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, S, time_points, tuple((param_vec,Cin)))[1:]

    pred_N = sol[-1, 0:2]

    return pred_N

def squared_loss(param_vec, current_S, Cin, next_N):
    predicted_N = predict(param_vec, current_S, Cin)

    return np.sum(np.array(next_N - predicted_N))**2

grad_func = grad(squared_loss)


f = open('./parameter_files/double_auxotroph.yaml')
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
param_vec = np.array([-0.2,-0.2, -0.2, -0.2, 50000., 50000., 50000., 50000., 2.5, 2.5, 0.0001, 0.000001, 0.00003, 0.00003])



labels = ['N1', 'N2', 'C1', 'C2', 'C0']

# load all system variables for two system trajectories
fullSol = np.load('fullSol.npy')
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


for i in range(len(fullSol)):
    Cin = Cins[i,:]
    sol = fullSol[i,:]

    sol1 = fullSol[i+1,:]
    next_N = sol1[:2]
    gradients = grad_func(param_vec, sol, Cin, next_N)
    param_vec -= gradients*0.0001
    print(param_vec)
