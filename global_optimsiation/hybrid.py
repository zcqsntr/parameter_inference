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
        self.personal_best_position = position
        self.personal_best_value = loss_function(*position)

    def update_velocity(self, global_best_position, c1, c2):
        R1 = np.random.random(self.position.shape)
        R2 = np.random.random(self.position.shape)

        cognitive = c1*R1*(self.personal_best_position - self.position)
        social = c2*R2*(global_best_position - self.position)

        self.velocity += cognitive + social

    def move_and_update(self, loss_function, domain):

        # if particle wont move out of bounds move it

        if ((domain[:,0]  < (self.position + self.velocity)).all() and  ((self.position + self.velocity) < domain[:,1]).all()):
            self.position += self.velocity
        else:
            self.velocity = -self.velocity
            self.position += self.velocity

        current_loss = loss_function(*self.position)

        if current_loss < self.personal_best_value:
            self.personal_best_value = current_loss
            self.personal_best_position = self.position

        return current_loss, self.position

    def step(self, loss_function, domain, global_best_position, c1, c2):
        self.update_velocity(global_best_position, c1, c2)
        loss, position = self.move_and_update(loss_function, domain)
        return loss, position


class Swarm():
    def __init__(self, loss_function, domain, n_particles, c1, c2):
        self.global_best_position = None
        self.global_best_value = None
        self.particles = []
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.loss_function = loss_function
        self.domain = domain
        self.initialise_particles(domain, n_particles)


    def initialise_particles(self, domain, n_particles):
        ''' sample postions within domain and put particles there'''

        # sample domain
        particles = []
        for _ in range(n_particles):
            # initialise particle
            position = np.random.uniform(domain[:,0], domain[:,1])
            particle = Particle(position, self.loss_function)
            particles.append(particle)

            # update gloabl loss
            loss = self.loss_function(*position)

            try:
                if loss < self.global_best_value:
                    self.global_best_value = loss
                    self.global_best_position = position
            except: # if None OK
                self.global_best_value = loss
                self.global_best_position = position

        self.particles = particles

    def step(self):
        for particle in self.particles:
            loss, position = particle.step(self.loss_function,self.domain, self.global_best_position, self.c1, self.c2)

            if loss < self.global_best_value: #OK
                self.global_best_value = loss
                self.global_best_position = position.copy()



    def find_minimum(self,n_steps):
        for _ in range(n_steps):
            self.step()
            print(self.global_best_value, self.global_best_position)
        return self.global_best_value, self.global_best_position




'''
sample particle from parameter priors, replace any outside of specified area
initialise particles
calculate global best and set it

for each time point
    for each particle
        update velocity
        update position
        evaluate fitnesses
        update personal and global best

        locally optimise a sub set of the particles, stop locally optimising if particle in minima

        remove particles in minima whos loss changes alot between steps as they are probably local minima


return global best solution
'''

x = np.linspace(-3,3,100)
y = np.linspace(-2,2,100)

X,Y = np.meshgrid(x,y)

Z = ackleys_function(X,Y)

fig, ax = plt.subplots()

#im = plt.contour(X,Y,Z, [-1.03, -1, -0.8, -0.7, -0.6, -0.3, 0,1,2,3, 4,5, 6, 7,8,10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 150, 200], vmin=abs(Z).min(), vmax=20)
im = plt.contour(X,Y,Z, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(im)

#plt.show()

domain = np.array([[-3, 3],[-2,2]])
n_particles = 1000

swarm = Swarm(six_hump_camel, domain, n_particles, 0.01, 0.01)
print(six_hump_camel(0.0898, -0.7126))
print(swarm.find_minimum(50))

# sample particles from region X[0] = [-3,3], X[1] = [-2,2]
