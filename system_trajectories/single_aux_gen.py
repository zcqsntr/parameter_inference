import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os
import yaml
import matplotlib
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))
from utilities import *

matplotlib.rcParams.update({'font.size': 22})

# open parameter file
f = open('../parameter_files/single_auxotroph.yaml')
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

T_MAX = 10000
A, num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds, gamma = Q_params

tSol = np.linspace(0, T_MAX, T_MAX+1)

count = 0
'''
for N1 in range(2,20,2):
    print(N1)
    for N2 in range(2,20,2):
        for action in range(100):
            initial_X = np.array([N1, N2])
            initial_C = np.array(param_dict['Q_params'][8])
            initial_C0 = param_dict['Q_params'][9]
            X = np.append(initial_X, initial_C)
            X = np.append(X, initial_C0)
            xSol = [X]
            Cin = action_to_state(action, 2, 10, [0, 1])
            for t in range(T_MAX):
                X = odeint(sdot2, X, [t, t+1], args=(Cin, A,ode_params, num_species))[-1]
                xSol.append(X)

                if (X[0] < 1/1000) or (X[1] < 1/1000):

                    break

                if t == T_MAX -1:
                    print('c')
                    count +=1



print('number: ', count)
'''

# set initial conditions


print('X:', initial_X)
print('C:', initial_C)
print('C0:',initial_C0)
X = np.append(initial_X, initial_C)
X = np.append(X, initial_C0)
xSol = np.array([X])
Cin = np.array([1, 0.1]).reshape(1, 2)


Cins = np.array()
time_diff = 3

for t in range(T_MAX):

    if t % time_diff == 0:
        Cin = np.random.randint(0,2, size = (1,2)) * np.random.randint(1,4, size = (1,2)) # choose random C0
    if X[0] < 10000:
        Cin[0][1] = 0
    else:
        Cin[0][1] = np.random.randint(1,4, size = (1,1))



    # get solution
    sol = odeint(sdot, X, [t + x *1 for x in range(time_diff)], args=(Cin, A,ode_params, num_species))[1:]


    X = sol[-1,:]
    xSol = np.append(xSol,sol, axis = 0)

    for _ in range(time_diff - 1):
        Cins = np.append(Cins, Cin, axis = 0)

    if (X[0] < 1/1000) or (X[1] < 1/1000):
        break

    if t == T_MAX -1:
        print('coexistence')

print(xSol.shape)
print(Cins.shape)
print(xSol)
'''
np.save('single_aux.npy', xSol)
np.save('single_aux_Cins.npy', Cins)
'''
# plot
plt.figure(figsize = (16.0,12.0))
xSol = np.array(xSol)
labels = ['N1', 'N2', 'C1', 'C2', 'C0']

for i in range(2):
    plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = labels[i])

plt.legend()
#plt.show()
