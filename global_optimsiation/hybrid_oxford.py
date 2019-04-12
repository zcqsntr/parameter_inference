import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from matplotlib import cm


from inference_utils import *
import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.animation as animation
import scipy as sp
from hybrid_noise_test import six_hump_camel, ackleys_function
from particle_swarm_optimiser import *

class OxfordSystem():
    def __init__(self, ode_params):
        self.ode_params = ode_params


    def sdot(self, S,t, params, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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
        '''
        # autograd gives t as an array_box, need to convert to int
        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t_ = t._value
            t_ = int(t)
        else:
            t_ = int(t)
        t = min(Cin.shape[0] - 1, t) # to prevent solver from going past the max time
        '''

        #Cin = Cins[int(t)]
        Cin = Cin.T
        C0in, q, y, y3, Rmax, Km, Km3, A = [np.array(param) for param in params]
        #Rmax = np.array(params[0:2])
        #Km = np.array(params[2:4])
        #Km3 = np.array(params[4:6])


        num_species = 2
        # extract variables
        N = np.array(S[:num_species])
        C = np.array(S[num_species:2*num_species])
        C0 = np.array(S[-1])

        try:
            R = self.monod(C, C0, Rmax, Km, Km3)
        except Exception as e:
            print(C.shape, C0.shape, Rmax.shape, Km.shape, Km3.shape)
            raise e

        Cin = Cin[:num_species]
        # calculate derivatives
        dN = N * (R.astype(float) + np.matmul(A,N) - q) # q term takes account of the dilution

        dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)

        dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))


        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC)
        dsol = np.append(dsol, dC0)



        return dsol

    def monod(self, C, C0, Rmax, Km, Km0):
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


        growth_rate = ((Rmax*C)/ (Km + C)) * (C0/ (Km0 + C0))

        return growth_rate

    def predict(self, params, S, Cins, time_points):
        '''
        predicts the populations at the next time point based on the current values for the params
        '''

        dt = time_points[1] - time_points[0]
        '''
        solver = sp.integrate.ode(self.sdot).set_integrator('lsoda', nsteps = 10)

        solver.set_initial_value(S, 0).set_f_params(params, Cin)
        sol = []
        sol.append([0, *S])
        i = 0

        # suppress Fortran-printed warning
        solver._integrator.iwork[2] = -1

        while len(sol) < len(time_points):

            solver.integrate(solver.t + dt)
            sol.append([solver.t, *solver.y])
            solver.set_initial_value(solver.y, solver.t)
            print(solver.t)

            if not solver.successful():
                print('FAILED AT: ')
                print(params)
                print()

        '''


        # get solution
        sol = odeint(self.sdot, S, time_points, args=(params, Cins), mxstep = 500)

        '''
        time_diff = 2
        xSol = np.array([S])
        for t in range(len(time_points)):

            Cin = Cins[t]
            # get solution
            sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin, self.ode_params, 2))[1:]

            S = sol[-1,:]
            xSol = np.append(xSol,sol, axis = 0)


        sol = np.array(xSol)
        '''

        pred_S = sol



        return pred_S

if __name__ == '__main__':
    true_params = np.array([480000, 480000, 520000, 520000, 1.8, 2, 0.00048776, 0.00000102115, 0.00006845928, 0.00006845928])
    prior_sigmas = np.array([100000., 100000., 100000., 100000., 1., 1., 0.005, 0.0001, 0.0001, 0.0001])
    likelihood_sigmas = np.array([1., 1.])
    prior_centres = np.array([490000, 470000, 510000, 530000, 2.1, 1.9, 0.0006, 0.0000012, 0.00007, 0.00006])
    # parameters to learn

    likelihood_scaling = np.array([1,1,1,1,1e12, 1e12, 1e12, 1e12, 1, 1, 0.1, 0.0001, 0.0001, 0.0001])


    f = open('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/examples/parameter_files/MPC.yaml')
    param_dict = yaml.load(f)
    f.close()

    validate_param_dict(param_dict)
    param_dict = convert_to_numpy(param_dict)

    ode_params = param_dict['ode_params']

    initial_X = param_dict['Q_params'][7]
    initial_C = param_dict['Q_params'][8]
    initial_C0 = param_dict['Q_params'][9]

    initial_S = np.append(initial_X, initial_C)
    initial_S = np.append(initial_S, initial_C0)

    labels = ['N1', 'N2', 'C1', 'C2', 'C0']


    def info(x,i,g):
        print('iter:', i)
        print('x:', x)
        print('g:', g)

    xSol = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/MPC_double_aux_rand.npy')
    Cins = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/MPC_double_aux_Cins_rand.npy')


    actual_N = xSol[0:20, 0:2]

    fullSol = xSol
    param_vecs = []
    square_losses = []
    MAP_losses = []
    param_losses = []
    centre_losses = []

    actual_params = param_dict['ode_params'][2:]

    domain = np.array([[400000, 600000], [400000, 600000], [400000, 600000], [400000, 600000]])
    domain = np.array([[1.5, 3.], [1.5, 3.], [0.0001, 0.001], [0.00000005, 0.0000005], [0.00001, 0.0001], [0.00001, 0.0001]])
    #domain = np.array([[0.0001, 0.001], [0.00000005, 0.0000005], [0.00001, 0.0001], [0.00001, 0.0001]])

    velocity_scaling = np.array([100000,100000,100000,100000,100000,1,1])
    n_particles = 50
    n_groups = 5
    cs = (0.1, 0.1)
    n_steps = 100
    system = OxfordSystem(param_dict['ode_params'])
    swarm = Swarm(system, domain, n_particles, n_groups, cs, ode_params)


    plt.figure()

    #plt.plot(time_points, xSol[:,0])
    #plt.show()
    x = np.linspace(domain[0][0], domain[0][1],100)
    y = np.linspace(domain[1][0], domain[1][1],100)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros(X.shape)



    '''
    for i in range(Z.shape[0]):
        print(i)
        for j in range(Z.shape[1]):
            Z[i,j] = swarm.squared_loss([X[i,j], Y[i,j]], xSol[0, :], Cins[:50],actual_N)

    #Z = np.load('oxford_umax_loss_func.npy')
    np.save('oxford_umax_loss_func.npy', Z)
    ind = Z.argmin()
    ind = np.unravel_index(ind, (100,100))
    min = np.min(Z)
    print(min)
    # 299888.86710959993 at 1.803030303030303 2.0

    print(Z)
    print(min)
    print(X[ind], Y[ind])

    fig = plt.figure(figsize = (12,8))
    im = plt.contour(X,Y,Z,40, vmin=abs(Z).min(), vmax=abs(Z).max())

    cb = fig.colorbar(im)
    plt.show()
    sys.exit()

    '''
    print(swarm.squared_loss([1.8, 2., 0.00048776, 0.000000102115,0.00006845928, 0.00006845928], xSol[0, :], Cins[:50],actual_N))

    swarm.find_minimum(initial_S, Cins[:10], actual_N , n_steps,'param')


    print(swarm.global_best_positions)
    print(swarm.global_best_values)

    fig = plt.figure(figsize = (12,8))
    plt.xlabel('u1_max')
    plt.ylabel('u2_max')
    plt.plot([1.8], [2.])
    ani = animation.ArtistAnimation(fig, swarm.ims, interval=50, blit=True,repeat_delay=1000)
    ani.save('oxford_TS.mp4', bitrate = 1000)
    plt.show()




#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000)
'''
for p in swarm.particles:
    im = plt.plot(*p.position, 'ro')
'''
#plt.show()
