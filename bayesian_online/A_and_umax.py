

import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))
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
    Rmax = np.array(Rmax)
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
def predict(A, N, Cin, C, C_0):

    # extract params from param_vec into form used by the rest of the code
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, N, time_points, tuple((A,Cin, C, C_0)))[1:]

    pred_N = sol[-1, 0:2]

    return pred_N

def predict_time_series(param_vec, initial_N, Cins, initial_C, initial_C0):
    time_points = np.array([x *1 for x in range(len(Cins))])
    sol = odeint(sdot, initial_N, time_points, tuple((param_vec,Cins, initial_C, initial_C0)))
    pred_Ns = sol[:, 0:2]


    return pred_Ns

def squared_loss(A, current_S, Cin, next_N, C, C_0):
    predicted_N = predict(A, current_S, Cin, C, C_0)

    return np.sum(np.array(next_N - predicted_N)**2)

def gaussian(xs, means, sigmas):
    return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-1/(2* sigmas**2)*(xs - means)**2)


def MAP_loss(param_vec, current_S, Cin , next_N, C, C_0, debug = False):
    predicted_N = predict(param_vec, current_S, Cin, C, C_0)

    priors = gaussian(param_vec, prior_centres, prior_sigmas) # centre on true params for now

    likelihoods = gaussian(next_N, predicted_N, likelihood_sigmas)

    if debug:
        print()
        print('predicted_N', predicted_N)
        print('next_N', next_N)
        print('priors:', priors)
        print('likelihoods: ', likelihoods)

    return  -np.sum(np.log(likelihoods)) -  1/weight * np.sum(np.log(priors))

def TS_square_loss(param_vec, initial_S, Cins, actual_Ns, initial_C, initial_C0, debug = False):
    predicted_Ns = predict_time_series(param_vec, initial_N, Cins, initial_C, initial_C0)
    print(predicted_Ns[0])
    print(actual_Ns[0])
    print(predicted_Ns.shape)
    print(actual_Ns.shape)
    return np.sum((predicted_Ns - actual_Ns)**2)

grad_func = grad(MAP_loss)

def grad_wrapper(param_vec, i):
    '''
    for the autograd optimisers
    '''
    return grad_func(param_vec, N, Cin, next_N, C, C_0)


TS_grad_func = grad(TS_square_loss)
def TS_grad_wrapper(param_vec, i):
    return TS_grad_func(param_vec, initial_N, Cins, actual_Ns, initial_C, initial_C0)

'''
xSol, Cins = create_time_series('/Users/Neythen/masters_project/app/CBcurl_master/examples/parameter_files/random.yaml', '/Users/Neythen/masters_project/results/lookup_table_results/auxotroph_section_10_repeats/WORKING/repeat0/Q_table.npy', 10000)

np.save('smaller_target.npy', xSol)
np.save('smaller_target_Cins.npy', Cins)

'''

f = open('../parameter_files/double_auxotroph.yaml')
param_dict = yaml.load(f)
f.close()


validate_param_dict(param_dict)
param_dict = convert_to_numpy(param_dict)

# extract parameters
NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, \
    MAX_STEP_SIZE, MIN_EXPLORE_RATE, cutoff, _, _  = param_dict['train_params']
NOISE, error = param_dict['noise_params']
num_species, num_controlled_species, num_N_states, num_Cin_states = \
    param_dict['Q_params'][1], param_dict['Q_params'][2], param_dict['Q_params'][3],param_dict['Q_params'][5]
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:8]
initial_N = param_dict['Q_params'][8]
initial_C = param_dict['Q_params'][9]
initial_C0 = param_dict['Q_params'][10]



prior_centres = np.array([-0.11, -0.09, -0.11, -0.09, 2.1, 1.9])
prior_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 1, 1])
likelihood_sigmas = np.array([1., 1.])

labels = ['N1', 'N2', 'C1', 'C2', 'C0']


xSol = np.load('../system_trajectories/double_aux.npy')
Cins = np.load('../system_trajectories/double_aux_Cins.npy')
fullSol = xSol


actual_Ns = fullSol[:, 0:2]


def info(x,i,g):
    print('iter:', i)
    print('x:', x)
    print('g:', g)


''' ONLINE'''

for weight in [1, 10, 100, 1000, 10000, 100000]:
    # parameters to learn
    param_vec = np.array([-0.2,-0.2, -0.2, -0.2, 1.5, 1.5])
    param_vecs = []
    square_losses = []
    MAP_losses = []
    param_losses = []
    centre_losses = []

    actual_A = Q_params[0]
    umax = ode_params[4]

    actual_params = np.append(actual_A.reshape(4,), umax)

    for i in range(500):
        alpha = get_learning_rate(i, 0.05,  0.5, 100) 
        Cin = Cins[i,:]
        sol = fullSol[i,:]
        N = sol[:2]
        C = np.array(sol[num_species:2*num_species])
        C_0 = np.array(sol[-1])

        sol1 = fullSol[i+1,:]
        next_N = sol1[:2]
        new_param_vec = adam(grad_wrapper, param_vec, num_iters = 10)
        #alpha = max(np.exp(-i/50), 0.00000001)

        param_vec = (1-alpha) * param_vec + alpha * new_param_vec
        param_vecs.append(param_vec)

        grads = grad_wrapper(param_vec, 0)
        print(grads)

        #param_vec -= grads*0.00001 * np.exp(-i/1000)
        M_loss = MAP_loss(param_vec, N, Cin, next_N, C, C_0, debug = True)
        MAP_losses.append(M_loss)

        s_loss = squared_loss(param_vec, N, Cin, next_N, C, C_0)
        square_losses.append(s_loss)

        param_loss = np.sum(np.abs(param_vec-actual_params)/np.abs(actual_params))
        param_losses.append(param_loss)

        centre_loss = np.sum(np.abs(param_vec-prior_centres)/np.abs(actual_params))
        centre_losses.append(centre_loss)

        print('weight: ', weight)
        print('iteration: ', i)
        print('MAP_loss: ', M_loss)
        print('Square losss', s_loss)
        print('params:', param_vec)
        print('param loss: ', np.sum((param_vec-actual_params)**2))

    print(centre_losses)
    print(param_losses)
    param_vecs = np.array(param_vecs)

    folder = 'regularisation_scan'

    plt.figure()
    plt.plot(param_losses, label = 'real parameters')
    plt.plot(centre_losses, label = 'prior centres')
    plt.title('Inferring A and u_max ' + str(weight))
    plt.xlabel('Iteration')
    plt.ylabel('Sum of square loss of the parameters')
    plt.legend()
    plt.savefig('./'+folder+'/parameter_loss_A' + str(weight) + '.png')

    plt.figure()
    plt.plot(MAP_losses, label = 'MAP loss')
    plt.title('Inferring A and u_max')
    plt.plot(square_losses, label = 'squared loss')
    plt.title('Inferring A and u_max ' + str(weight))
    plt.xlabel('Iteration')
    plt.ylabel('Loss in predicted N')
    plt.legend()
    plt.savefig('./'+folder+'/N_loss_A_and_umax' + str(weight) + '.png')

    plt.figure()
    [a11, a12, a21, a22,umax1, umax2] = plt.plot(param_vecs)
    plt.legend([a11, a12, a21, a22,umax1, umax2], ['a11', 'a12', 'a21', 'a22','umax1', 'umax2'])
    plt.title('param evolution')
    plt.xlabel('Iteration')
    plt.ylabel('params')
    plt.savefig('./'+folder+'/param_evolution' + str(weight) + '.png')



'''TIME SERIES'''
'''
new_param_vec = adam(TS_grad_wrapper, param_vec, callback = info, num_iters = 5000)
print(new_param_vec)
'''
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
