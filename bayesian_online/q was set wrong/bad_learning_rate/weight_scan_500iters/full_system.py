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
file to estimate all parameters using autograd
'''

def sdot(S, t, param_vec, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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

    y = param_vec[4:6]
    y3 = param_vec[6:8]
    Rmax = param_vec[8:10]
    Km = true_params[10:12]
    Km3 = true_params[12:14]


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
    growth_rate = ((Rmax*C)/ (Km + C)) * (C0/ (Km0 + C0))

    return growth_rate


def likelihood_objective(observed_dN, param_vec):
    pass


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''
def predict(params, S, Cin):
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, S, time_points, tuple((params, Cin)))[1:]

    pred_N = sol[-1, 0:2]

    return pred_N

def squared_loss(A, current_S, Cin, next_N):
    C = current_S[num_species: 2*num_species]
    C_0 = current_S[-1]
    predicted_N = predict(A, current_S, Cin)

    return np.sum(np.array(next_N - predicted_N)**2)

def gaussian(xs, means, sigmas):

    return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-1/(2* sigmas**2)*(xs - means)**2)

def box_prior(xs, means, sigmas):

    xs_take_means = xs - means
    xs_take_means[np.where(np.abs(xs_take_means) > 1)] = 0
    return (xs_take_means/sigmas)  # centred on mean, width sigma probably want to scale this so integral is 1



true_params = np.array([-0.1, -0.1, -0.1, -0.1, 480000, 480000, 520000, 520000, 2, 2, 0.00049, 0.00000102115, 0.00006845928, 0.00006845928])
prior_sigmas = np.array([1., 1., 1., 1., 100000., 100000., 100000., 100000., 2., 2., 0.005, 0.0001, 0.0001, 0.0001])
likelihood_sigmas = np.array([1., 1.])
prior_centres = np.array([-0.11, -0.09, -0.11, -0.09, 490000, 470000, 510000, 530000, 2.1, 1.9, 0.0006, 0.0000012, 0.00007, 0.00006])
# parameters to learn
param_vec = np.array([-0.9,-0.11, -0.11, -0.9, 500000., 500000., 500000., 500000., 2.1, 2.1, 0.0004, 0.000002, 0.00009, 0.00009])
prior_scaling = np.array([1,1,1,1,1e6, 1e6, 1e6, 1e6, 10, 10, 0.1, 0.0001, 0.0001, 0.0001])


def MAP_loss(param_vec, current_S, Cin, next_N, debug = False):
    C = current_S[num_species: 2*num_species]
    C_0 = current_S[-1]

    predicted_N = predict(param_vec, current_S, Cin)

    priors = gaussian(param_vec, prior_centres, prior_sigmas) # centre on true params for now

    likelihoods = gaussian(next_N, predicted_N, likelihood_sigmas)

    if debug:

        print('predicted_N', predicted_N)
        print('next_N', next_N)
        print('priors:',np.sum(np.log(priors)))
        print('likelihoods: ', -np.sum(np.log(likelihoods)) )


    return  -np.sum(np.log(likelihoods))  - 1/weight*np.sum(np.log(priors))




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
initial_X = param_dict['Q_params'][8]
initial_C = param_dict['Q_params'][9]
initial_C0 = param_dict['Q_params'][10]



labels = ['N1', 'N2', 'C1', 'C2', 'C0']


grad_func = grad(MAP_loss)

def grad_wrapper(param_vec, i):
    '''
    for the autograd optimisers
    '''

    return grad_func(param_vec, sol, Cin, next_N)

'''
xSol, Cins = create_time_series('/Users/Neythen/masters_project/app/CBcurl_master/examples/parameter_files/random.yaml', '/Users/Neythen/masters_project/results/lookup_table_results/auxotroph_section_10_repeats/WORKING/repeat0/Q_table.npy', 10000)

np.save('smaller_target.npy', xSol)
np.save('smaller_target_Cins.npy', Cins)

'''




for weight in [1, 10, 100, 1000, 10000, 100000]:
    xSol = np.load('../system_trajectories/double_aux.npy')
    Cins = np.load('../system_trajectories/double_aux_Cins.npy')
    print(xSol.shape)

    fullSol = xSol
    square_losses = []
    MAP_losses = []
    param_losses = []
    centre_losses = []

    actual_A = Q_params[0]
    other_params = np.array(ode_params[2:])

    actual_params = np.append(actual_A.reshape(4,), other_params.reshape(10, ))
    alpha = np.abs(param_vec)

    print(weight)
    for i in range(500):
        Cin = Cins[i,:]
        sol = fullSol[i,:]
        N = sol[:2]
        C = np.array(sol[num_species:2*num_species])
        C_0 = np.array(sol[-1])

        sol1 = fullSol[i+1,:]
        true_next_N = sol1[:2]

        next_N = true_next_N #+ np.random.normal(scale = 1., size = (2,))

        new_param_vec = adam(grad_wrapper, param_vec, num_iters = 10)
        #alpha = max(np.exp(-i/50), 0.00000001)

        #alpha = 0.5

        param_vec = (1-alpha) * param_vec + alpha * new_param_vec
        grads = grad_wrapper(param_vec, 0)
        print('grads: ', grads)
        '''
        param_vec -= grads*0.00001 * np.exp(-i/1000)
        '''


        M_loss = MAP_loss(param_vec, sol, Cin, next_N, debug = True)
        MAP_losses.append(M_loss)

        s_loss = squared_loss(param_vec, sol, Cin, next_N)
        square_losses.append(s_loss)

        param_loss = np.sum(np.abs(param_vec-actual_params)/np.abs(actual_params))
        param_losses.append(param_loss)

        centre_loss = np.sum(np.abs(param_vec-prior_centres)/np.abs(actual_params))
        centre_losses.append(centre_loss)

        print()
        print('Iteration: ', i)
        print('MAP_loss: ', M_loss)
        print('Square losss', s_loss)
        print('params:', param_vec)
        print('param loss: ', param_loss)

    plt.figure()
    plt.plot(param_losses, label = 'real parameters')
    plt.plot(centre_losses, label = 'prior centres')
    plt.title('Inferring everything but Ks ' + str(weight))
    plt.xlabel('Iteration')
    plt.ylabel('Sum of absolute loss of the inferred parameters')
    plt.legend()
    plt.savefig('parameter_loss_A' + str(weight) + '.png')



    plt.figure()
    plt.plot(MAP_losses, label = 'MAP loss')
    plt.title('Inferring A and u_max')
    plt.savefig('N_loss_A_and_umax.png')
    plt.plot(square_losses, label = 'squared loss')
    plt.title('Inferring everything but Ks ' + str(weight))
    plt.xlabel('Iteration')
    plt.ylabel('Loss in predicted N')
    plt.legend()
    plt.savefig('N_loss_A_and_umax' + str(weight) + '.png')



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
