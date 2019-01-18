from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
from autograd.scipy.integrate import odeint
from utilities import *

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

    Km = np.array([0.00049, 0.00000102115])
    Km3 = np.array([0.00006845928, 0.00006845928])
    '''
    Km = param_vec[10:12]
    Km3 = param_vec[12:14]
    '''
    num_species = 2
    # extract variables
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    C0in, q = np.array([1., 1.5])

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




def run_model(params):
    time_diff = 5  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])
    Cin = np.array([0.1, 0.1]) # initial value
    S = np.array([10., 10., 0.1, 0.1, 1.])
    sol = odeint(sdot, S, time_points, tuple((params, Cin)))[1:]

    pred_N = sol[-1, 0:2]

    return pred_N[0]

# full probelm

problem = {
    'num_vars': 15,
    'names': ['K_10', 'K_11', 'K_20', 'K_22', 'mu1_max', 'mu2_max', 'a_11', 'a_12', 'a_21', 'a_22',
        'q', 'y_11', 'y_22', 'y_10', 'y_20'],
        'bounds': [[0.00006, 0.00008],
                   [0.0004, 0.0006],
                   [0.00006, 0.00008],
                   [0.0000007, 0.0000013],
                   [1.8, 2.2],
                   [1.8, 2.2],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [1.3, 1.7],
                   [510000, 530000],
                   [510000, 530000],
                   [470000, 490000],
                   [470000, 490000]]
}
'''
problem = {
    'num_vars': 15,
    'names': ['K_10', 'K_11', 'K_20', 'K_22', 'mu1_max', 'mu2_max', 'a_11', 'a_12', 'a_21', 'a_22',
        'q', 'y_11', 'y_22', 'y_10', 'y_20'],
        'bounds': [[0.00001, 0.0001],
                   [0.0001, 0.001],
                   [0.00001, 0.0001],
                   [0.0000001, 0.000005],
                   [1., 3.],
                   [1., 3.],
                   [-0.15, -0.05],
                   [-0.15, -0.05],
                   [-0.15, -0.05],
                   [-0.15, -0.05],
                   [1., 2.],
                   [450000, 550000],
                   [450000, 550000],
                   [400000, 520000],
                   [400000, 520000]]
}

# without Ks
problem = {
    'num_vars': 11,
    'names': ['mu1_max', 'mu2_max', 'a_11', 'a_12', 'a_21', 'a_22',
        'q', 'y_11', 'y_22', 'y_10', 'y_20'],
        'bounds': [[1.8, 2.2],
                   [1.8, 2.2],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [-0.11, -0.09],
                   [1.3, 1.7],
                   [510000, 530000],
                   [510000, 530000],
                   [470000, 490000],
                   [470000, 490000]]
}
'''


param_values = saltelli.sample(problem, 10000)
print(param_values.shape)
# run model with all the parameter values
# do N1 first
N1s = np.zeros([param_values.shape[0]])

for i, p in enumerate(param_values):
    if i %10000== 0:
        print(i)
    N1s[i] = run_model(p)

Si = sobol.analyze(problem, N1s, print_to_console = True)
