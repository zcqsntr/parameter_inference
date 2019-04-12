import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os
import yaml
import matplotlib
import scipy as sp
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/CBcurl')
from utilities import *

matplotlib.rcParams.update({'font.size': 22})

# open parameter file
f = open('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/examples/parameter_files/MPC.yaml')
param_dict = yaml.load(f)
f.close()

validate_param_dict(param_dict)
param_dict = convert_to_numpy(param_dict)

# extract parameters
NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, \
    MAX_STEP_SIZE, MIN_EXPLORE_RATE, MAX_EXPLORE_RATE, cutoff, hidden_layers, buffer_size  = param_dict['train_params']
NOISE, error = param_dict['noise_params']
num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds = \
    param_dict['Q_params'][0], param_dict['Q_params'][1],  param_dict['Q_params'][2], param_dict['Q_params'][7], param_dict['Q_params'][4], param_dict['Q_params'][5]
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:7]
initial_X = param_dict['Q_params'][7]
initial_C = param_dict['Q_params'][8]
initial_C0 = param_dict['Q_params'][9]

T_MAX = 1000
num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds, gamma = Q_params

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

X = np.append(initial_X, initial_C)
X = np.append(X, initial_C0)
xSol = np.array([X])
Cin = np.array([0.1,0.1]).reshape(1, 2)

Cins = np.array(Cin)
time_diff = 2



sol  = []
def solout(t,y):
    sol.append([t,*y])


solver = sp.integrate.ode(sdot_object).set_integrator('lsoda', nsteps = 10000)

solver.set_initial_value(X, 0).set_f_params(Cin, ode_params, num_species)
sol = []
sol.append([0, *X])
i = 0
dt = 1

# suppress Fortran-printed warning
solver._integrator.iwork[2] = -1

while solver.t < T_MAX:
    print(i)
    i += 1
    solver.integrate(solver.t + dt)
    sol.append([solver.t, *solver.y])
    solver.set_initial_value(solver.y, solver.t)

xSol = np.array(sol)

#sol = odeint(sdot, X, [t + x *1 for x in range(time_diff)], args=(Cin, ode_params, num_species))[1:]

print(xSol.shape)


# plot
plt.figure(figsize = (16.0,12.0))
xSol = np.array(xSol)
labels = ['N1', 'N2', 'C1', 'C2', 'C0']
for i in range(5):
    plt.plot(xSol[:,0], xSol[:,i+1], label = labels[i])

plt.legend()
plt.show()
