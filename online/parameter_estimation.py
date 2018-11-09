import numdifftools as nd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.integrate as spi
from scipy.integrate import odeint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

import tensorflow as tf
from utilities import *
#from agents import *



'''
for t in range(T_MAX):
    X = odeint(sdot2, X, [t, t+1], args=(Cin, A,ode_params, num_species))[-1]
    xSol.append(X)

    if (X[0] < 1/1000) or (X[1] < 1/1000):
        break
'''



def dN1(S, t, C0in, C1in, q, y32, params, num_species):

    N1 = np.array(S[0])
    N2 = np.array(S[1])
    C1 = np.array(S[2])
    C2 = np.array(S[3])
    C0 = np.array(S[4])

    y, y3, Rmax, Km, Km3, a1, a2 = params

    R = monod2(C1, C0, Rmax, Km, Km3)

    dN = N1 * (R + a1*N1 + a2*N2 - q) # q term takes account of the dilution

    dC1 = q*(C1in - C1) - (1/y)*R*N1 # sometimes dC.shape is (2,2)

    dC0 = q*(C0in - C0) - 1/y3*R*N1 - 1/y32*R2*N2
    dC0 = np.array([dC0])
    sol = np.append(dN, dC)
    sol = np.append(sol, dC0)

    return sol


def sdot2(S, t, Cin, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    C0in, q, y, y3, Rmax, Km, Km3 = params

    R = monod(C, C0, Rmax, Km, Km3)
    #Cin = Cin[:num_species]
    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)
    dC0 = np.array([dC0])
    sol = np.append(dN, dC)
    sol = np.append(sol, dC0)
    return tuple(sol)

def sdot2_swapped(t, S, Cin, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    C0in, q, y, y3, Rmax, Km, Km3 = params

    R = monod2(C, C0, Rmax, Km, Km3)
    #Cin = Cin[:num_species]
    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)
    dC0 = np.array([dC0])
    sol = np.append(dN, dC)
    sol = np.append(sol, dC0)
    return sol


def N1(fit_params, S, ode_params, Cin, A, num_species):

    y1, y31, Rmax1, Km1, Km31= fit_params # N1 specific parameters to be fitted
    a1, a2 = -0.1, -0.11

    C0in, q, y, y3, Rmax, Km, Km3 = ode_params # params for all bacteria
    # set N1 params

    A[0][0], A[0][1] = a1, a2
    y[0], y3[0], Rmax[0], Km[0], Km3[0]  = y1, y31, Rmax1, Km1, Km31

    ode_params = C0in, q, y, y3, Rmax, Km, Km3
    time_diff = 4
    N1 = odeint(sdot2, S, [t + x *1 for x in range(time_diff)], args=(Cin,A, ode_params, num_species))[-1][0]
    '''ode = spi.ode(sdot2_swapped)
    ode.set_integrator('vode', nsteps = 500, method = 'bdf')
    ode.set_initial_value(S, 0)
    ode.set_f_params(Cin, A, ode_params, num_species)
    #for t in range(time_diff):
    ode.integrate(ode.t + time_diff)

    N1 = ode.y[0]

    '''
    return N1



def N1_squared_loss(fit_params, S, ode_params, Cin, A, num_species, actual_N1):

    pred_N1 = N1(fit_params, S, ode_params, Cin, A, num_species)
    if abs(pred_N1-actual_N1) < 0.0001:
        pass
        #print('pred: ', pred_N1)
        #print('actual: ', actual_N1)
    return (pred_N1 - actual_N1)**2

def adam(intital, n_iters = 1000, alpha = 0.001, b1 = 0.9, b2 = 0.9, eps = 10**-8):

    grads = np.array(nd.Gradient(N1_squared_loss)(fit_params, S, ode_params, Cin, A, num_species, actual_N1))

def monod_LV(N, C, q, C0, Ks, Ks0, umax, A):
    u = monod(C, C0, umax, Ks, Ks0)
    dN = N*(np.matmul(A, N) + u - q)

    return dN



'''
# SIMULATION CONSTANTS
tf.reset_default_graph() #Clear the Tensorflow graph.

num_species, num_x_states, num_C0_states = param_dict['Q_params'][1], param_dict['Q_params'][2],param_dict['Q_params'][4]

layer_sizes = [num_x_states**num_species] + hidden_layers + [num_C0_states**num_species]

agent = NeuralAgent(layer_sizes)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
'''

f = open('./parameter_files/double_auxotroph.yaml')
param_dict = yaml.load(f)
f.close()


validate_param_dict(param_dict)
param_dict = convert_to_numpy(param_dict)


NUM_EPISODES, test_freq, explore_denom, step_denom, MIN_TEMP, MAX_TEMP, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE, hidden_layers= param_dict['train_params']
NOISE, error = param_dict['noise_params']

ode_params = param_dict['ode_params']

#extract parameters
NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE,\
    MAX_STEP_SIZE, MIN_EXPLORE_RATE, cutoff, hidden_layers, buffer_size = param_dict['train_params']
NOISE, error = param_dict['noise_params']
num_species, num_controlled_species, num_N_states, num_Cin_states = \
    param_dict['Q_params'][1], param_dict['Q_params'][2],  param_dict['Q_params'][3],param_dict['Q_params'][5]
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:8]
initial_X = param_dict['Q_params'][8]
initial_C = param_dict['Q_params'][9]
initial_C0 = param_dict['Q_params'][10]

# initial guess for parameters
fit_params = np.array([1000000., 1000000., 2.8, 2.1, 1.9])
A, num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds, gamma = Q_params
alpha = 0.001

print(A)

Q_table = np.load('/Users/Neythen/masters_project/results/lookup_table_results/spock_14_N1_10_8>N2>4_new_gammas/WORKING/WORKING_saved_Q_table/Q_table.npy')
X = initial_X
C = initial_C
C0 = initial_C0
S = np.append(X, C)
S = np.append(S, C0)
xSol = np.array([S])
print(S.shape)
Cins = np.array([0,0]).reshape(1,2)




labels = ['N1', 'N2', 'C1', 'C2', 'C0']
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,i], label = labels[i])
plt.legend()

np.save('fullSol2', xSol)
np.save('Cins2', Cins)







# ESTIMATES N1 SPECIFIC PARAMETERS
'''
xSol = np.load('fullSol.npy')
Cins = np.load('Cins.npy')

for _ in range(100):
    for t in range(1000 - 1):

        S = xSol[t,:]

        actual_N1 = xSol[t+1,0]
        Cin = Cins[t,:]
        grads = np.array(nd.Gradient(N1_squared_loss)(fit_params, S, ode_params, Cin, A, num_species, actual_N1))

        fit_params -= alpha*grads

        if t%100 == 0:
            print(t)
            print('grads: ', grads)
            print('FPs: ',fit_params)

print('grads: ', grads)
print('FPs: ',fit_params) #y1, y31, Rmax1, Km1, Km31= fit_params

'''

N1_params = [1000000., 1000000., 2.9, 2., 2., 0.1, 0.1]
params = [5., 0.1, np.array([1000000., 1000000.]), np.array([1000000., 1000000]), np.array([2.9,3.]), np.array([2.,2.]), np.array([2.,2.])]
A = np.array([[-0.1, 0.11],[0.1,0.1]])


grad = nd.Gradient(N1_squared_loss)(N1_params, np.array([1.,1.,1.,1.,1.]), params, 5., A, 2, 1.3)
print(grad)



xSol = np.load('/Users/Neythen/masters_project/results/lookup_table_results/spock_N3/WORKING/WORKING_data/LTPops.npy')

plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,0])
plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,1])




fullSol = np.load('fullSol.npy')
fullSol2 = np.load('fullSol2.npy')
Ks = np.array(ode_params[5])
Ks0 = np.array(ode_params[6])
umax = np.array(ode_params[4])
A = np.array(Q_params[0])

Ksf = np.array([1.6, 2.6])
Ks0f = np.array([1.6, 2.6])
umaxf = np.array([2.6, 3.6])
Af = np.array([[-0.09,-0.09], [-0.1,-0.09]])

alpha = 0.0001
q = ode_params[1]

labels = ['N1', 'N2', 'C1', 'C2', 'C0']
'''
plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol[:,0])), fullSol[:,i], label = labels[i])
plt.legend()

plt.figure()
for i in range(5):
    plt.plot(np.linspace(0,T_MAX,len(fullSol2[:,0])), fullSol2[:,i], label = labels[i])
plt.legend()
plt.show()
'''




''' Old way of doing it with dN objective works, but C0 doesnt vary enough for Ks0 to be founds'''
'''
print('S: ', S)
print('A: ', Af)
print('umax: ', umaxf)
print('Ksf: ', Ksf)
print('Ks0: ', Ks0f)


for _ in range(999999999999):
    loss = 0
    for t in range(T_MAX):
        S = fullSol[t, :]

        N = S[:2]
        C = S[2:4]
        C0 = S[4]
        pred_dN = monod_LV(N, C, q, C0, Ks, Ks0, umax, Af)
        actual_dN = monod_LV(N, C, q, C0, Ks, Ks0, umax, A) # replace this with dN estimate from timesteps

        # get the first part of the gradient descent step
        deltas = 2*(pred_dN - actual_dN)


        # make full gradient descent step matrix
        dA = np.array([N*N[0]*deltas[0], N*N[1]*deltas[1]])

        dKs = -2*N*(pred_dN - actual_dN) *umaxf* C*C0 / ((Ksf+C)**2 *(Ks0f+C0))
        dKs0 = -2*N*(pred_dN - actual_dN) *umaxf* C*C0 / ((Ksf+C)* (Ks0f+C0)**2)
        dumax = 2*N*(pred_dN - actual_dN) * C0/(Ks0f + C0) * C/(Ks+C)



        Af -= alpha * dA
        Ks0f -= alpha * dKs0
        Ksf -= alpha* dKs
        umaxf -= alpha * dumax
        loss += (pred_dN - actual_dN)**2


    for t in range(T_MAX):
        S = fullSol2[t, :]

        N = S[:2]
        C = S[2:4]
        C0 = S[4]
        pred_dN = monod_LV(N, C, q, C0, Ksf, Ks0, umaxf, Af)
        actual_dN = monod_LV(N, C, q, C0, Ks, Ks0, umax, A) # replace this with dN estimate from timesteps

        # get the first part of the gradient descent step
        deltas = 2*(pred_dN - actual_dN)

        # make full gradient descent step matrix
        dA = np.array([N*N[0]*deltas[0], N*N[1]*deltas[1]])

        dKs = -2*N*(pred_dN - actual_dN) *umaxf* C*C0 / ((Ksf+C)**2 *(Ks0f+C0))
        dKs0 = -2*N*(pred_dN - actual_dN) *umaxf* C*C0 / ((Ksf+C)* (Ks0f+C0)**2)
        dumax = 2*N*(pred_dN - actual_dN) * C0/(Ks0f + C0) * C/(Ks+C)


        Af -= alpha * dA
        Ks0f -= alpha * dKs0
        Ksf -= alpha* dKs
        umaxf -= alpha * dumax

        loss += (pred_dN - actual_dN)**2

    print('S: ', S)
    print('A: ', Af)
    print('umax: ', umaxf)
    print('Ksf: ', Ksf)
    print('Ks0: ', Ks0f)
    print('loss: ', loss/2000)

'''
