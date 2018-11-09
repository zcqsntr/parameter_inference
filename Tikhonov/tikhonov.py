

import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import *
from inference_utils import *
import numpy as np



# FIRST DO SIMPLE CHEMOSTAT SYSTEM, assuming growth rate u kknown at each timestep
# this gets rid of the non-linearities, deal with that later. Also assuming Cs known at each timestep


f = open('../parameter_files/single_auxotroph.yaml')
param_dict = yaml.load(f)
f.close()

xSol = np.load('../system_trajectories/single_aux.npy')
Cins = np.load('../system_trajectories/single_aux_Cins.npy')

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

N1s = xSol[:,0]
N2s = xSol[:,1]
# skip irrelevant auxotrophic nutrient for non auxotrophic bacteria
CTs = xSol[:, 3]
C0s = xSol[:,4]

CTins = Cins[:,1]


deltaN1s = np.array([N1s[i+1] - N1s[i] for i in range(len(N1s) - 1)])
deltaN2s = np.array([N2s[i+1] - N2s[i] for i in range(len(N2s) - 1)])
deltaC0s = np.array([C0s[i+1] - C0s[i] for i in range(len(C0s) - 1)])
deltaCTs = np.array([CTs[i+1] - CTs[i] for i in range(len(CTs) - 1)])

#remove last datapoints in trajectories to make same size as deltas, same thing done in the paper
N1s = N1s[:-1]
N2s = N2s[:-1]
CTs = CTs[:-1]
C0s = C0s[:-1]
CTins = CTins[:-1]
C0ins = np.full(CTins.shape, ode_params[0])

# make growth rate arrays
Rmax = ode_params[4]
Km = ode_params[5]
Km0 = ode_params[6]

growth_rates = []

for i in range(len(CTins)):
    C = xSol[i, 2:4]
    C0 = xSol[i, 4]

    growth_rates.append(monod(C, C0, Rmax, Km, Km0))

growth_rates = np.array(growth_rates)


u1s = growth_rates[:, 0]
u2s = growth_rates[:,1]


u1N1s = u1s*N1s
u2N2s = u2s*N2s

# F = PY

# create F
F = np.array([deltaC0s, deltaCTs, deltaN1s, deltaN2s])

print('F shape:', F.shape)

#create Y

Y = np.array([C0ins, CTins, C0s, CTs, N1s, N2s, u1N1s, u2N2s])

print('Y shape: ', Y.shape)

reg = 999999

denom = np.matmul(Y, Y.T) + reg
print(denom.shape)
num = np.matmul(F, Y.T)

P = np.matmul(num, np.linalg.inv(denom))
print("P shape: ", P.shape)
print(P)

q = ode_params[1]
y01 = ode_params[2][0]
y02 = ode_params[2][1]
yT = ode_params[3][1]


actual_P = np.array([[q,0,-q,0,0,0,-1/y01,-1/y02],
                     [0,q,0,-q,0,0,0,-1/yT],
                     [0,0,0,0,-q,0,1,0],
                     [0,0,0,0,0,-q,0,1]])


print(actual_P)

print(np.linalg.norm(P-actual_P))

for i in range(4):
    print(np.linalg.norm(P[i,:]-actual_P[i,:]))



# Reduced system just with the ys

F = np.array([deltaC0s - q*(C0ins - C0s),
              deltaCTs - q*(CTins - CTs)])


Y = np.array([u1N1s,
              u2N2s])

print('Y shape: ', Y.shape)

reg = 9

denom = np.matmul(Y, Y.T) + reg

num = np.matmul(F, Y.T)

print("F shape: ", F.shape)
P = np.matmul(num, np.linalg.inv(denom))
print("P shape: ", P.shape)
print(P)

'''

from sklearn.linear_model import Ridge


clf = Ridge(alpha = 10, fit_intercept = False, max_iter = 100)
clf.fit(Y.T, F.T)
print("RR: ",clf.coef_)
print(actual_P)

coefs = np.array(clf.coef_)

for i in range(4):
    print()
    print(np.linalg.norm(P[i,:]-actual_P[i,:]))
    print(np.linalg.norm(coefs[i,:]-actual_P[i,:]))




from sklearn.linear_model import Lasso
clf = Lasso(alpha = 99, fit_intercept = False)
clf.fit(Y.T, F.T)
print("LASSO: ",clf.coef_)
print(clf.intercept_)
'''
