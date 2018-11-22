import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import cm

from utilities import *
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


def six_hump_camel(X,Y):
    '''
    test function for optimisation algorithms
    domain:
        X[0] = [-3,3], X[1] = [-2,2]
    minima:
        f(X*) = -1.0316 at X* = +-[0.0898, -0.7126]
    '''
    return (4 - 2.1*X**2 + X**4/3)*X**2 + X*Y + (-4 + 4*Y**2)*Y**2

def ackleys_function(X,Y):
    ''' domian x1,x2 = [-40,40]'''
    firstSum = X**2.0 + Y**2.0
    secondSum = np.cos(2.0*math.pi*X) + np.cos(2.0*math.pi*Y)
    return -20.0*np.exp(-0.2*np.sqrt(firstSum/2)) - np.exp(secondSum/2) + 20 + np.e

class Particle():
    def __init__(self, position, loss_function):

        self.position = position
        self.velocity = np.zeros(position.shape)
        self.stuck = False
        R1 = np.random.random()
        R2 = np.random.random()
        R3 = np.random.random()
        R4 = np.random.random()

        self.personal_best_position = position
        self.personal_best_value = loss_function(*position) + self.noise(np.array([0.08986, -0.71268]), 0.3, position, R1, R2, R3, R4)

    def update_velocity(self, global_best_position, c1, c2):
        '''
        updates every particles velocity based on the cognitive and social components
        '''

        R1 = np.random.random(self.position.shape)
        R2 = np.random.random(self.position.shape)

        cognitive = c1*R1*(self.personal_best_position - self.position)
        social = c2*R2*(global_best_position - self.position)

        self.velocity = 0.7*self.velocity + cognitive + social

    def move_and_update(self, loss_function, domain, R1, R2, R3, R4):
        '''
        moves particles and updates best values and positions
        '''
        # if particle wont move out of bounds move it to a random point in the bounds as in the hybrid paper
        if ((domain[:,0]  < (self.position + self.velocity)).all() and  ((self.position + self.velocity) < domain[:,1]).all()):
            self.position += self.velocity
        else:
            self.position = np.random.uniform(domain[:,0], domain[:,1])


        true_minima = np.array([0.08986, -0.71268])

        current_loss = loss_function(*self.position) + self.noise(true_minima, 0.3, self.position, R1, R2, R3, R4)

        '''
        # if is close to personal minima but loss funciton is changing then in a noisy place and not the global
        if np.linalg.norm(self.position - self.personal_best_position) < 0.001 and abs(current_loss - self.personal_best_value) > 0.3:
            self.position = np.random.uniform(domain[:,0], domain[:,1])
            self.velocity = np.zeros(self.position.shape)
            current_loss = loss_function(*self.position) + self.noise(true_minima, 0.3, self.position, R1, R2, R3, R4)
            self.personal_best_value = current_loss
            self.personal_best_position = self.position

        '''


        if current_loss < self.personal_best_value:
            self.personal_best_value = current_loss
            self.personal_best_position = self.position.copy()

        return current_loss, self.position

    def gradient_descent(self, loss_func, grad_func):
        '''
        does the gradient descent step
        '''
        position = adam(grad_func, self.position, num_iters = 10)
        loss = loss_func(*position)
        if loss < self.personal_best_value:
            self.personal_best_value = loss
            self.personal_best_position = position.copy()

        return loss, position

    def step(self, loss_function, grad_func, domain, global_best_position, c1, c2, R1, R2, R3, R4):
        '''
        does one step for a particle. Updates velocity then moves by PSO and GD
        '''
        self.update_velocity(global_best_position, c1, c2)
        loss, position = self.move_and_update(loss_function, domain, R1, R2, R3, R4)

        loss, position = self.gradient_descent(loss_function, grad_func)

        return loss, position

    def noise(self,true_minima, radius, position, R1, R2, R3, R4):

        if np.linalg.norm(true_minima - position) > radius: # add noise
            noise = R1/2*(np.cos(5*R3*position[0]) + 1) + R2/2*(np.sin(5*R4*position[1]) + 1)
        else:
            noise = 0

        noise = 0
        return noise

    def get_total_loss(self,loss_function, radius, position, true_minima, R1, R2, R3, R4):

        noise = self.noise(true_minima, radius, position, R1, R2, R3, R4)

        loss = loss_function(*position) + noise

        return loss

class Swarm():
    def __init__(self, loss_function, grad_func, domain, n_particles, n_groups, c1, c2):
        self.global_best_positions = [None] * n_groups
        self.global_best_values = [None] * n_groups
        self.particles = []
        self.c1 = c1
        self.c2 = c2
        self.loss_function = loss_function
        self.domain = domain
        self.grad_func = grad_func
        self.initialise_particles(domain, n_particles, n_groups)

    def initialise_particles(self, domain, n_particles, n_groups):
        ''' sample postions within domain and puts particles there'''

        # sample domain
        particles = []

        for i in range(n_groups):
            group = []

            for _ in range(n_particles):
                # initialise particle
                position = np.random.uniform(domain[:,0], domain[:,1])
                particle = Particle(position, self.loss_function)
                group.append(particle)

                # update gloabl loss
                loss = self.loss_function(*position)

                try:
                    if loss < self.global_best_values[i]:
                        self.global_best_values[i] = loss
                        self.global_best_positions[i] = position.copy()
                except: # if None OK
                    self.global_best_values[i] = loss
                    self.global_best_positions[i] = position.copy()
            particles.append(group)

        self.particles = particles

    def step(self):
        '''
        carries out one step of the algorithm for all particles
        '''

        colours = ['ro', 'bo', 'go', 'mo', 'ko']
        group_xs = []
        group_ys = []
        R1 = np.random.random()
        R2 = np.random.random()
        R3 = np.random.random()
        R4 = np.random.random()
        for i,group in enumerate(self.particles):
            x = []
            y = []

            for particle in group:
                true_minima = np.array([0.08986, -0.71268])
                particle_loss = particle.get_total_loss(self.loss_function, 0.3, particle.position, true_minima, R1, R2, R3, R4)

                if np.linalg.norm(particle.position - self.global_best_positions[i]) < 0.01 and abs(particle_loss - self.global_best_values[i]) > 0.1:

                    particle.position = np.random.uniform(domain[:,0], domain[:,1])

                    particle.personal_best_position = particle.position
                    self.global_best_positions[i] = particle.position

                    current_loss = self.loss_function(*particle.position) + particle.noise(true_minima, 0.3, particle.position, R1, R2, R3, R4)
                    self.global_best_values[i] = current_loss
                    particle.personal_best_value = current_loss

                loss, position = particle.step(self.loss_function, self.grad_func, self.domain, self.global_best_positions[i], self.c1, self.c2, R1, R2, R3, R4)

                if loss < self.global_best_values[i]:
                    self.global_best_values[i] = loss
                    self.global_best_positions[i] = position.copy()

                x.append(particle.position[0])
                y.append(particle.position[1])

            group_xs.append(x)
            group_ys.append(y)

        # add current frame to plot
        plotting_data = []

        for i in range(len(group_xs)):
            plotting_data.append(group_xs[i])
            plotting_data.append(group_ys[i])
            plotting_data.append(colours[i])


        im = plt.plot(*plotting_data)

        ims.append(im)

    def find_minimum(self,n_steps):
        '''
        runs the hybrid PSO/gradient descent alogirhtm and returns the minima found
        '''
        for _ in range(n_steps):
            self.step()

        return self.global_best_values, self.global_best_positions


if __name__ == '__main__':
    # plot noise function
    x = np.linspace(-3,3,100)
    y = np.linspace(-2,2,100)
    '''
    x = np.linspace(-40,40,100)
    y = np.linspace(-40,40,100)
    '''
    X,Y = np.meshgrid(x,y)

    Z = np.zeros(X.shape)

    R1 = np.random.random()
    R2 = np.random.random()
    R3 = np.random.random()
    R4 = np.random.random()

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = Particle.noise(None, np.array([0.08986, -0.71268]), 0.3, np.array([X[i,j], Y[i,j]]), R1, R2, R3, R4)

    fig, ax = plt.subplots()


    im = plt.contour(X,Y,Z, vmin=abs(Z).min(), vmax=abs(Z).max())

    #cb = fig.colorbar(im)


    x = np.linspace(-3,3,100)
    y = np.linspace(-2,2,100)
    '''
    x = np.linspace(-40,40,100)
    y = np.linspace(-40,40,100)
    '''
    X,Y = np.meshgrid(x,y)


    function = six_hump_camel
    Z = function(X,Y)

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize = (16,12))

    im = plt.contour(X,Y,Z, [-1.03, -1, -0.8, -0.7, -0.6, -0.3, 0,1,2,3, 4,5, 6, 7,8,10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 150, 200], vmin=abs(Z).min(), vmax=20)
    #im = plt.contour(X,Y,Z, vmin=abs(Z).min(), vmax=abs(Z).max())
    cb = fig.colorbar(im)
    ims = []

    domain = np.array([[-3, 3],[-2,2]])
    #domain = np.array([[-40, 40],[-40,40]])
    n_particles = 10
    n_groups = 5

    grad_func = grad(function)

    def grad_wrapper(param_vec, i):
        '''
        for the autograd optimisers
        '''

        return grad_func(*param_vec)

    swarm = Swarm(function, grad_wrapper, domain, n_particles, n_groups, 0.2, 0.2)

    print(swarm.find_minimum(500))

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000)
    ani.save('hybrid.mp4', bitrate = 1000)
    '''
    for p in swarm.particles:
        im = plt.plot(*p.position, 'ro')
    '''
    plt.show()

    # sample particles from region X[0] = [-3,3], X[1] = [-2,2]
