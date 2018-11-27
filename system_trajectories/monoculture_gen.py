import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os
import yaml
import matplotlib
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import *

matplotlib.rcParams.update({'font.size': 22})

# open parameter file
f = open('../parameter_files/monoculture.yaml')
param_dict = yaml.load(f)
f.close()


ode_params = param_dict['ode_params']
initial_X = param_dict['Q_params'][6]
initial_C0 = param_dict['Q_params'][7]

T_MAX = 1000

tSol = np.linspace(0, T_MAX, T_MAX+1)
count = 0


# set initial conditions

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
    q, y, Rmax, Km = params


    R = monod(C0, Rmax, Km)

    # calculate derivatives
    dN = N * (R.astype(float) - q) # q term takes account of the dilution

    dC0 = q*(C0in - C0) - 1/y*R*N

    # consstruct derivative vector for odeint
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC0)
    return tuple(dsol)


print('X:', initial_X)
print('C0:',initial_C0)

X = np.append(initial_X, initial_C0)
xSol = np.array([X])


Cins = np.array([1.])
time_diff = 20
Cin = 1
for t in range(T_MAX):

    if t == 500:
        Cin = 0
    if t % time_diff == 0:
        #Cin = np.random.randint(0,2) # choose random C0
        Cin += np.random.randint(-2,2) * 0.01
        print(Cin)

    # get solution
    sol = odeint(sdot, X, [t + x *1 for x in range(time_diff)], args=(Cin,ode_params))[1:]

    X = sol[-1,:]
    xSol = np.append(xSol,sol, axis = 0)

    for _ in range(time_diff - 1):
        Cins = np.append(Cins, np.array([Cin]), axis = 0)

    if (X[0] < 1/1000):
        break

    if t == T_MAX -1:
        print('coexistence')

print(xSol.shape)
print(Cins.shape)
print(xSol)
'''
np.save('monoculture.npy', xSol)
np.save('monoculture_Cins.npy', Cins)
'''
# plot
plt.figure(figsize = (16.0,12.0))
xSol = np.array(xSol)
labels = ['N1', 'C0']

for i in range(2):
    plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = labels[i])

plt.legend()
plt.show()
