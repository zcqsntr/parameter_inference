import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import cm
from utilities import *

'''
import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
'''
import numpy as np
from scipy.integrate import odeint
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.animation as animation

import copy


class Particle():
    '''
    Particle to be used in PSO
    '''
    def __init__(self, position, cs):

        self.position = copy.deepcopy(position)
        #self.velocity = np.array([np.random.random() * 5000, np.random.random()*0.005])
        self.velocity = np.zeros(position.shape)

        self.personal_best_position = copy.deepcopy(position)
        self.personal_best_value = None
        self.cs = cs

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps

    def evaluate(self, loss_function, initial_S, constant, target, mode):
        #current_loss = loss_function(self.position, sol, Cin, next_N) # CHANGE FOR PARMETER INFERENCE/MPC
        if mode == 'MPC':
            Cin = self.position
            parameters = constant
        else:
            Cin = constant
            parameters = self.position

        current_loss = loss_function(parameters, initial_S, Cin, target) # Cin is actually parameters, self.position is Cin, for MPC

        '''
        # if is close to personal minima but loss funciton is changing then in a noisy place and not the global
        if np.linalg.norm(self.position - self.personal_best_position) < 0.001 and abs(current_loss - self.personal_best_value) > 0.3:
            self.position = np.random.uniform(domain[:,0], domain[:,1])
            self.velocity = np.zeros(self.position.shape)
            current_loss = loss_function(*self.position) + self.noise(true_minima, 0.3, self.position, R1, R2, R3, R4)
            self.personal_best_value = current_loss
            self.personal_best_position = self.position

        '''
        #update swarm values based on new particles position
        try:
            if current_loss < self.personal_best_value:
                self.personal_best_value = copy.deepcopy(current_loss)
                self.personal_best_position = copy.deepcopy(self.position)
        except:
            self.personal_best_value = copy.deepcopy(current_loss)
            self.personal_best_position = copy.deepcopy(self.position)

        return current_loss

    def inertial_weight(self, t):
        x = 1 - t/self.n_steps
        if x > 0.9:
            return 0.9
        elif x < 0.4:
            return 0.4
        else:
            return x


    def update_velocity(self, global_best_position, t):
        '''
        updates every particles velocity based on the cognitive and social components
        '''
        c1, c2 = self.cs

        R1 = np.random.random(self.position.shape)
        R2 = np.random.random(self.position.shape)

        cognitive = c1*R1*(self.personal_best_position - self.position)
        social = c2*R2*(global_best_position - self.position)

        self.velocity = (self.inertial_weight(t) * self.velocity + cognitive + social)

    def update_position(self, domain):
        '''
        moves particles and updates best values and positions
        '''
        # if particle wont move out of bounds move it to a random point in the bounds as in the hybrid paper
        if ((domain[:,0]  < (self.position + self.velocity)).all() and ((self.position + self.velocity) < domain[:,1]).all()):
            self.position += self.velocity
        else:
            self.position = np.random.uniform(domain[:,0], domain[:,1])


        return self.position
    '''
    def gradient_descent(self, grad_wrapper):
    '''
        #does the gradient descent step
    '''
        position = adam(grad_wrapper, self.position, num_iters = 10)
        if ((domain[:,0]  < position).all() and (position < domain[:,1]).all()):
            self.position = copy.deepcopy(position)
        else:
            self.position = np.random.uniform(domain[:,0], domain[:,1])

        return self.position
    '''

    def step(self, loss_function, grad_wrapper, domain,  global_best_position, constant, current_S, target, mode, t):
        '''
        does one full hybrid step for a particle. Updates velocity then moves by PSO and GD
        '''

        self.update_velocity(global_best_position, t)
        position = self.update_position(domain)
        #position = self.gradient_descent(loss_function, grad_wrapper, domain, Cin, sol, actual_N)
        loss = self.evaluate(loss_function, current_S, constant, target, mode)

        return loss, position

class Swarm():
    '''
    Swarm for PSO. Manages a swarm of particles
    '''
    def __init__(self, domain, n_particles, n_groups, cs, ode_params = None, loss_function = None):
        self.ode_params = ode_params
        self.global_best_positions = [None] * n_groups
        self.global_best_values = [None] * n_groups
        self.particles = []
        if loss_function is None:
            self.loss_function = self.MAP_loss
        else:
            self.loss_function = loss_function
        #self.grad_func = grad(self.loss_function)
        self.domain = domain
        self.ims = [] # for plotting

        self.initialise_particles(domain, n_particles, n_groups, cs)

    def grad_wrapper(self, param_vec, i):
        '''
        for the autograd optimisers
        '''

        actual_N = self.fullSol[0:50,0]

        return self.grad_func(param_vec, self.fullSol[0,:], self.Cins, actual_N)

    def initialise_particles(self, domain, n_particles, n_groups, cs):
        '''
        sample postions within domain and puts particles there
        '''

        particles = []

        # each group of particles
        for i in range(n_groups):
            group = []

            #each particle
            for _ in range(n_particles):
                # initialise particle
                position = np.random.uniform(domain[:,0], domain[:,1])

                particle = Particle(position, cs)
                group.append(particle)

            particles.append(group)

        self.particles = particles

    def reset_particle(self, particle, domain, i): #maybe refactor to be a method of Particle
        '''
        resets particles position when particle is at a noisy (local) minima
        '''
        particle.position = np.random.uniform(domain[:,0], domain[:,1])
        particle.personal_best_position = copy.deepcopy(particle.position)
        self.global_best_positions[i] = copy.deepcopy(particle.position) # helps group leave local minima
        current_loss = self.loss_function(particle.position, self.sol, self.Cin, self.next_N)
        self.global_best_values[i] = copy.deepcopy(current_loss)
        particle.personal_best_value = copy.deepcopy(current_loss)

    def MAP_loss(self, param_vec, current_S, Cin, target,  debug = False):
        '''
        loss functions using liklihoods and priors
        '''

        time_points = np.array([t for t in range(len(target) + 1)]) # +1 for the initial point

        predicted_N = self.predict(param_vec, current_S, Cin, time_points)[:,:2]

        #priors = self.gaussian(param_vec, prior_centres, prior_sigmas) # centre on true params for now
        likelihood_sigma = 3000.


        likelihood = self.gaussian(target, predicted_N, likelihood_sigma)
        if debug:
            print('predicted_N', predicted_N)
            print('next_N', N)
            print('priors:',np.sum(np.log(priors)))
            print('likelihoods: ', np.sum(np.log(likelihoods)) )

        return  -np.sum(np.log(likelihood))  #- 0*1/weight*np.sum(np.log(priors))

    def gaussian(self, xs, means, sigmas):
        '''
        returns the probability density at a point according to a gaussian distribution
        '''

        return 1/np.sqrt(2*np.pi*sigmas**2) * np.sum(np.exp(-1/(2* sigmas**2)*(xs - means)**2))

    def squared_loss(self, param_vec, current_S, Cin, target,  debug = False): # verified working on timeseries
        '''
        squared loss
        '''
        if self.mode == 'MPC':
            param_vec = constant
            Cin = variable
        else:
            Cin = constant
            param_vec = variable

        num_species = 2
        C = current_S[num_species: 2*num_species]
        C_0 = current_S[-1]
        time_points = np.array([t for t in range(50)])#PUT THIS BACK FOR ONLINE

        predicted_N = self.predict_time_series(param_vec, current_S, Cin, time_points)[:,0]

        return np.sum(np.array(actual_N - predicted_N)**2)

    def sdot(self, S, t, params, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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

        # autograd gives t as an array_box, need to convert to int
        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t = t._value
            t = int(t)
        else:
            t = int(t)
        t = min(Cin.shape[0] - 1, t) # to prevent solver from going past the max time

        C0in = Cin[t]

        #C0in = Cin
        N = S[0]
        C0 = S[1]
        # extract parameters
        q = self.ode_params[0]
        y, Rmax = params[0:2]
        Km =  self.ode_params[3]
        R = self.monod(C0, Rmax, Km)

        # calculate derivatives
        dN = N * (R.astype(float) - q) # q term takes account of the dilution
        dC0 = q*(C0in - C0) - 1/y*R*N

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC0)

        return tuple(dsol)

    def monod(self, C0, Rmax, Km0):
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

    def predict(self, params, S, Cin, time_points):

        sol = odeint(sdot, S, time_points, tuple((Cin, params, 2)))[1:] #PUT THIS BACK FOR ONLINE
        return sol
    '''
    def predict_time_series(self, params, S, Cins, time_points): # verified working on timeseries

        sol = odeint(self.sdot, S, time_points, tuple((params, Cins)))[:] #PUT THIS BACK FOR ONLINE

        pred_N = sol[:, 0] #PUT THIS BACK FOR ONLINE

        return pred_N
    '''
    def step(self, current_S, constant, target, mode, t):
        '''
        carries out one step of the algorithm for all particles
        '''
        colours = ['ro', 'bo', 'go', 'mo', 'ko'] # for plotting
        group_xs = []
        group_ys = []

        for i,group in enumerate(self.particles):
            x = []
            y = []

            for particle in group:
                '''
                particle_loss = self.loss_function(particle.position, sol, Cin, actual_N)

                # if stuck in a noisey minima
                if np.linalg.norm((particle.position - self.global_best_positions[i])*np.array([10, 0.000001])) < 0.01 and abs(particle_loss - self.global_best_values[i]) > 10:
                    self.reset_particle(particle, self.domain, i)
                '''

                loss, position = particle.step(self.loss_function, self.grad_wrapper, self.domain, self.global_best_positions[i], constant, current_S, target, mode, t)

                # if particle has found a new best place
                if loss < self.global_best_values[i]:
                    self.global_best_values[i] = copy.deepcopy(loss)
                    self.global_best_positions[i] = copy.deepcopy(position)
                '''
                x.append(particle.position[0])
                y.append(particle.position[1])
                '''
            '''
            group_xs.append(x)
            group_ys.append(y)
            '''

        '''
        # add current frame to plot
        plotting_data = []
        for i in range(len(group_xs)):
            plotting_data.append(group_xs[i])
            plotting_data.append(group_ys[i])
            plotting_data.append(colours[i])

        im = plt.plot(*plotting_data)
        self.ims.append(im)
        '''

    def find_minimum(self, initial_S, constant, target, n_steps, mode):

        # iniitlise loss at initial positions
        for i,group in enumerate(self.particles):
            for particle in group:
                particle.set_n_steps(500)
                loss = particle.evaluate(self.loss_function, initial_S, constant, target, mode)

                #update swarm values based on new particles position
                try:
                    if loss < self.global_best_values[i]:
                        self.global_best_values[i] = copy.deepcopy(loss)
                        self.global_best_positions[i] = copy.deepcopy(particle.position)
                except:
                    self.global_best_values[i] = copy.deepcopy(loss)
                    self.global_best_positions[i] = copy.deepcopy(particle.position)


        for i in range(n_steps):

            self.step(initial_S, constant, target, mode, i)

        return self.global_best_values, self.global_best_positions, self.ims

    def sdot_co(self, S, t, param_vec, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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
        '''
        A = param_vec[5]
        #A = param_vec[0]
        y = param_vec[0]
        y3 = param_vec[1]

        Rmax = param_vec[2]

        Km = self.ode_params[5]
        Km3 = self.ode_params[6]

        Km = param_vec[10:12]
        Km3 = param_vec[12:14]
        '''

        # autograd gives t as an array_box, need to convert to int
        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t = t._value
            t = int(t)
        else:
            t = int(t)
        t = min(Cin.shape[0] - 1, t) # to prevent solver from going past the max time

        Cin = Cin[t]

        print(" param vec: ", param_vec)
        A = np.reshape(param_vec[-4:], (2,2))
        y = param_vec[4:6]
        y3 = param_vec[6:8]

        Rmax = param_vec[8:10]

        Km = self.ode_params[5]
        Km3 = self.ode_params[6]

        num_species = 2
        # extract variables
        N = np.array(S[:num_species])
        C = np.array(S[num_species:2*num_species])
        C0 = np.array(S[-1])

        C0in, q = self.ode_params[:2]

        R = self.monod_co(C, C0, Rmax, Km, Km3)

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

    def monod_co(self, C, C0, Rmax, Km, Km0):
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
    def predict_co(self, params, S, Cin):
        '''
        predicts the populations at the next time point based on the current values for the params
        '''
        time_diff = 2  # frame skipping
        time_points = np.array([x *1 for x in range(time_diff)])
        sol = odeint(self.sdot, S, time_points, tuple((params, Cin)))[1:]
        pred_N = sol[-1, 0:2]

        return pred_N

    """
    def find_minimum_online(self, full_sol, Cins, n_steps):
        '''
        runs the hybrid PSO/gradient descent alogirhtm and returns the minima found
        '''
        num_species = 2
        for i in range(n_steps):
            print(i)
            Cin = Cins[i]
            current_S = full_sol[i,:]

            sol1 = full_sol[i+1,:]

            next_N = sol1[0]

            # for co culture
            '''
            self.Cin = self.Cins[i,:]
            self.sol = self.fullSol[i,:]

            N = sol[:2]
            C = np.array(sol[num_species:2*num_species])
            C_0 = np.array(sol[-1])

            sol1 = self.fullSol[i+1,:]

            self.next_N = sol1[:2]
            '''
            self.step(current_S, Cin, next_N)

        return self.global_best_values, self.global_best_positions, self.ims

    def find_minimum_time_series(self, full_sol, Cins, n_steps):
        '''
        runs the hybrid PSO/gradient descent alogirhtm and returns the minima found
        '''
        num_species = 2
        initial_S =
        for i in range(n_steps):
            self.step(fullSol[0,:], Cins[0:50], actual_N)

        return self.global_best_values, self.global_best_positions, self.ims

    def find_minimum_MPC(self, current_S, n_steps):

        for i in range(n_steps):
            self.step(current_S, self.parameters, self.target) # self.Cins is parameters for MPC atm

        return self.global_best_values, self.global_best_positions, self.ims


    def sdot_co(self, S, t, param_vec, Cin): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
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

        Km = self.ode_params[5]
        Km3 = self.ode_params[6]
        '''
        Km = param_vec[10:12]
        Km3 = param_vec[12:14]
        '''
        num_species = 2
        # extract variables
        N = np.array(S[:num_species])
        C = np.array(S[num_species:2*num_species])
        C0 = np.array(S[-1])

        C0in, q = self.ode_params[:2]

        R = self.monod(C, C0, Rmax, Km, Km3)

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
    def monod_co(self, C, C0, Rmax, Km, Km0):
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
    def predict_co(self, params, S, Cin):
        '''
        predicts the populations at the next time point based on the current values for the params
        '''
        time_diff = 2  # frame skipping
        time_points = np.array([x *1 for x in range(time_diff)])
        sol = odeint(self.sdot, S, time_points, tuple((params, Cin)))[1:]
        pred_N = sol[-1, 0:2]

        return pred_N
    """
