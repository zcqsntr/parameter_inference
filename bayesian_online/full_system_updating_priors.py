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
from scipy.optimize import curve_fit


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
    #A = param_vec[0]

    y = param_vec[4:6]
    y3 = param_vec[6:8]

    Rmax = param_vec[8:10]


    Km = ode_params[5]
    Km3 = ode_params[6]
    '''
    Km = param_vec[10:12]
    Km3 = param_vec[12:14]
    '''


    num_species = 2
    # extract variables
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    C0in, q = ode_params[:2]


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
    '''
    returns the probablity density of at xs from a gaussian defined by means and sigmas
    '''

    return 1/np.sqrt(2*np.pi*sigmas**2) * np.exp(-1/(2* sigmas**2)*(xs - means)**2)

def box_prior(xs, means, sigmas):

    xs_take_means = xs - means
    xs_take_means[np.where(np.abs(xs_take_means) > 1)] = 0
    return (xs_take_means/sigmas)  # centred on mean, width sigma probably want to scale this so integral is 1


def sample_output(S, Cin,param_means, param_sigmas):
    '''
    samples the current distribution over parameters to estimate the corresponding distribution over outputs
    '''
    output_samples = []

    for _ in range(10000):
        params = np.random.normal(param_means, param_sigmas)
        output = predict(params, S, Cin)
        output_samples.append(output)

    return np.array(output_samples)

def sample_params(param_means, param_sigmas, n):
    param_samples = []

    for _ in range(n):
        params = np.random.normal(param_means, param_sigmas)

        param_samples.append(params)


    return np.array(param_samples).reshape(n,14)


def fit_gaussian(S, Cin, param_means, param_sigmas):
    '''
    generates predictions using parameters sampled form the distribution, then outputs
    a gaussian fitted to the predictions
    '''

    output_samples = sample_output(sol, Cin,true_params, prior_sigmas)*100

    # generate approximate gaussian
    bins = np.array(range(1000))

    indeces = np.digitize(output_samples, bins)

    count1 = np.bincount(indeces[:,0], minlength = 1001)[1:]
    print(bins.shape)
    print(count1.shape)

    print(count1.shape)
    print(count1)
    count2 = np.bincount(indeces[:,1], minlength = 1001)[1:]

    # fit gaussians
    '''
    popt1 = curve_fit(gaussian, bins, count1)
    popt2 = curve_fit(gaussian, bins, count2)


    print('popt1', popt1)
    print('popt2', popt2)
    '''
    plt.plot(bins,count1)
    plt.plot(bins,count2)

    plt.show()
    sys.exit()

'''
def fit_gaussian(data):
    opt, cov = curve_fit(gaussian, )

'''


true_params = np.array([-0.1, -0.1, -0.1, -0.1, 480000, 480000, 520000, 520000, 2, 2, 0.00049, 0.00000102115, 0.00006845928, 0.00006845928])
prior_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 100000., 100000., 100000., 100000., 1., 1., 0., 0., 0., 0.])
likelihood_sigmas = np.array([1., 1.])
prior_centres = np.array([-0.11, -0.09, -0.11, -0.09, 490000, 470000, 510000, 530000, 2.1, 1.9, 0.0006, 0.0000012, 0.00007, 0.00006])
# parameters to learn

likelihood_scaling = np.array([1,1,1,1,1e12, 1e12, 1e12, 1e12, 1, 1, 0.1, 0.0001, 0.0001, 0.0001])




def MAP_loss(param_vec, current_S, Cin, next_N, debug = False):
    C = current_S[num_species: 2*num_species]
    C_0 = current_S[-1]

    predicted_N = predict(param_vec, current_S, Cin)

    #priors = gaussian(param_vec, prior_centres, prior_sigmas) # centre on true params for now

    likelihoods = gaussian(next_N, predicted_N, likelihood_sigmas)

    if debug:

        print('predicted_N', predicted_N)
        print('next_N', next_N)
        print('priors:',np.sum(np.log(priors)))
        print('likelihoods: ', np.sum(np.log(likelihoods)) )

    return  -np.sum(np.log(likelihoods))  #- 0*np.sum(np.log(priors))




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
    param_dict['Q_params'][1], param_dict['Q_params'][2], param_dict['Q_params'][3],param_dict['Q_params'][5]
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:8]
initial_X = param_dict['Q_params'][8]
initial_C = param_dict['Q_params'][9]
initial_C0 = param_dict['Q_params'][10]

labels = ['N1', 'N2', 'C1', 'C2', 'C0']


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



learning_scaling = np.array([0.1,0.1,0.1,0.1,500000,500000,1000000,1000000, 1, 1,0.0001, 0.0000001, 0.00001, 0.00001])
'''
#for weight in [1,10,100,1000,10000,100000]:
for weight in [1]:
    param_vec = true_params

    xSol = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux.npy')
    Cins = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux_Cins.npy')
    print(xSol[0])
    print(xSol.shape)
    fullSol = xSol
    param_vecs = []
    square_losses = []
    MAP_losses = []
    param_losses = []
    centre_losses = []

    actual_A = Q_params[0]
    other_params = np.array(ode_params[2:])

    actual_params = np.append(actual_A.reshape(4,), other_params.reshape(10, ))

    for i in range(100):

        alpha = get_learning_rate(i, 0.05,  0.5, 100) * learning_scaling

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
        fit_gaussian(sol, Cin, true_params, prior_sigmas)

        #param_vec = (1-alpha) * param_vec + alpha * new_param_vec
        grads = grad_wrapper(param_vec, 0)


        #param_vec -= grads*0.000000000001 * np.exp(-i/1000)


        direction = new_param_vec - param_vec
        param_vec += alpha * direction

        param_vecs.append(param_vec)


        M_loss = MAP_loss(param_vec, sol, Cin, next_N, debug = True)
        MAP_losses.append(M_loss)

        s_loss = squared_loss(param_vec, sol, Cin, next_N)
        square_losses.append(s_loss)

        param_loss = np.sum(np.abs(param_vec-actual_params)/np.abs(actual_params))
        param_losses.append(param_loss)

        centre_loss = np.sum(np.abs(param_vec-prior_centres)/np.abs(actual_params))
        centre_losses.append(centre_loss)

        print('weight: ', weight)
        print('Iteration: ', i)
        print('MAP_loss: ', M_loss)
        print('Square losss', s_loss)
        print('params:', param_vec)
        print('param loss: ', param_loss)
        print('grads: ', grads)
        print()

    param_vecs = np.array(param_vecs)
    folder = 'no_k'

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
    [a11, a12, a21, a22] = plt.plot(param_vecs[:,:4])
    plt.legend([a11, a12, a21, a22], ['a11', 'a12', 'a21', 'a22'])
    plt.title('A evolution')
    plt.xlabel('Iteration')
    plt.ylabel('params')
    plt.savefig('./'+folder+'/A_evolution' + str(weight) + '.png')

    plt.figure()
    [umax1, umax2] = plt.plot(param_vecs[:,8:10])
    plt.legend([umax1, umax2], ['umax1', 'umax2'])
    plt.title('Umax evolution')
    plt.xlabel('Iteration')
    plt.ylabel('params')
    plt.savefig('./'+folder+'/umax_evolution' + str(weight) + '.png')

    plt.figure()
    [y1, y2, y11, y22] = plt.plot(param_vecs[:,4:8])
    plt.legend([y1, y2, y11, y22], ['y1', 'y2', 'y11', 'y22'])
    plt.title('y evolution')
    plt.xlabel('Iteration')
    plt.ylabel('params')
    plt.savefig('./'+folder+'/y_evolution' + str(weight) + '.png')
    plt.show()

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


# get behaviour of loss function at different points in parameter space

# define distribution to sample from, dont sampe Ks as solver problems

true_params = np.array([-0.1, -0.1, -0.1, -0.1, 480000, 480000, 520000, 520000, 2, 2, 0.00049, 0.00000102115, 0.00006845928, 0.00006845928])
sigmas = np.array([0.1, 0.1, 0.1, 0.1, 10000., 10000., 10000., 10000., 1., 1., 0., 0., 0., 0.])

sampled_parameters = sample_params(true_params, sigmas, 10)

# include true params in sampled points
sampled_parameters = np.append(sampled_parameters, true_params.reshape(1,14), axis = 0)

param_vec = true_params

xSol = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux.npy')
Cins = np.load('/Users/Neythen/Desktop/masters_project/parameter_estimation/system_trajectories/double_aux_Cins.npy')

fullSol = xSol

TMAX = 100
n_param_sets = sampled_parameters.shape[0]


MAP_losses = np.zeros((n_param_sets, TMAX))
square_losses = np.zeros((n_param_sets, TMAX))

for i in range(TMAX):

    Cin = Cins[i,:]
    sol = fullSol[i,:]
    N = sol[:2]
    C = np.array(sol[num_species:2*num_species])
    C_0 = np.array(sol[-1])

    sol1 = fullSol[i+1,:]
    true_next_N = sol1[:2]

    next_N = true_next_N #+ np.random.normal(scale = 1., size = (2,))

    for j in range(n_param_sets):
        param_vec = sampled_parameters[j,:]
        M_loss = MAP_loss(param_vec, sol, Cin, next_N)
        MAP_losses[j,i] = M_loss
        s_loss = squared_loss(param_vec, sol, Cin, next_N)
        square_losses[j,i] = s_loss


for i in range(n_param_sets - 1):
    plt.plot(square_losses[i,:])
plt.plot(square_losses[-1,:], label = 'true params', color = 'black')
plt.legend()
plt.xlabel('time')
plt.ylabel('MAP_loss')
plt.ylim([-10,200])

plt.show()
