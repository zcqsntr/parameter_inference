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
def monod(C0, Rmax, Km0):
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


def sdot(S, t, C0in, params): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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
    # extract variables
    N = S[0]
    C0 = S[1]


    # extract parameters
    q = ode_params[0]
    y, Rmax, Km = params
    Km =  ode_params[3]


    R = monod(C0, Rmax, Km)

    # calculate derivatives
    dN = N * (R.astype(float) - q) # q term takes account of the dilution

    dC0 = q*(C0in - C0) - 1/y*R*N

    # consstruct derivative vector for odeint
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC0)
    return tuple(dsol)


'''
param_vec needs to be a row/column vector so need to reshape the parameters,
then change them back

'''
def predict(params, N, Cin):
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(sdot, N, time_points, tuple((Cin, params)))[1:]

    pred_N = sol[-1, 0:2]

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



learning_scaling = np.array([10000000,1, 0.00001])

#for weight in [1,10,100,1000,10000,100000]:
for weight in [1]:
    param_vec = np.array([500000., 0.8, 0.00045])

    xSol = np.load('/Users/Neythen/masters_project/parameter_estimation/system_trajectories/monoculture.npy')
    Cins = np.load('/Users/Neythen/masters_project/parameter_estimation/system_trajectories/monoculture_Cins.npy')
    print(xSol[0])
    print(xSol.shape)
    fullSol = xSol
    param_vecs = []
    square_losses = []
    MAP_losses = []
    param_losses = []
    centre_losses = []

    actual_params = ode_params[1:]
    for _ in range(100):
        for i in range(30):

            alpha = get_learning_rate(i, 0.05,  0.5, 100) * learning_scaling

            Cin = Cins[i]
            sol = fullSol[i,:]
            N = np.array(sol[0])
            C_0 = np.array(sol[1])

            sol1 = fullSol[i+1,:]
            true_next_N = sol1[0]

            next_N = true_next_N #+ np.random.normal(scale = 1., size = (2,))

            new_param_vec = adam(grad_wrapper, param_vec, num_iters = 10)
            #alpha = max(np.exp(-i/50), 0.00000001)

            #alpha = 0.5

            #param_vec = (1-alpha) * param_vec + alpha * new_param_vec
            grads = grad_wrapper(param_vec, 0)
            '''

            param_vec -= grads*0.000000000001 * np.exp(-i/1000)
            '''

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
