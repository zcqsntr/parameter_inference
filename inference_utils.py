
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.integrate as spi
from scipy.integrate import odeint

sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(ROOT_DIR)


from utilities import *

def create_time_series(param_path, Q_table_path, n_points):
    Q_table = np.load(Q_table_path)

    f = open(param_path)
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

    A, num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds, gamma = Q_params



    X = initial_X
    C = initial_C
    C0 = initial_C0

    S = np.append(X, C)
    S = np.append(S, C0)
    xSol = np.array([S])

    Cins = np.array([0,0]).reshape(1,2)


    for t in range(n_points):

        state = np.array(state_to_bucket(X, N_bounds, num_N_states))
        action_indeces = np.unravel_index(np.argmax(Q_table[tuple(state)]), Q_table[tuple(state)].shape)
        action_index = np.ravel_multi_index(action_indeces, [num_Cin_states, num_Cin_states])

        # turn action index into C0
        Cin = action_to_state(action_index, num_species, num_Cin_states, Cin_bounds) # take out this line to remove the effect of the algorithm

        Cins = np.append(Cins, Cin.reshape(1,2), axis = 0)
        time_diff = 4

        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin,A,ode_params, num_species))

        X1 = sol[-1, :num_species]

        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]
        actual_N1 = X1[0]

        # if the below block is uncommented it behaves very strangley, one species dies out quickly,which shouldnt happend at all
        if t %100 == 0:
            print(t)
            print('--------------------------', X1)

        X = X1
        C = C1
        C0 = C01

        S = np.append(X1, C1)

        S = np.append(S, C01)

        xSol = np.append(xSol, sol.reshape(-1, 5), axis = 0)





    return xSol, Cins
