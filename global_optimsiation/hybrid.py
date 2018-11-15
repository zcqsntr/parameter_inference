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

def six_hump_camel(X,Y):
    '''
    test function for optimisation algorithms
    domain:
        X[0] = [-3,3], X[1] = [-2,2]
    minima:
        f(X*) = -1.0316 at X* = +-[0.0898, -0.7126]
    '''
    return (4 - 2.1*X**2 + X**4/3)*X**2 + X*Y + (-4 + 4*Y**2)*Y**2


class Particle():
    def __init__(self, ):
        self.personal_best_value = None
        self.global_best_value = None
        self.position = None
        self.velocity = None
        self.stuck = False


class Swarm():
    def __init__(self, ):
        self.personal_best_position = None
        self.global_best_position = None
        self.particles = []


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




x = np.linspace(-2,2,100)
y = np.linspace(-1,1,100)

X,Y = np.meshgrid(x,y)

Z = six_hump_camel(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

plt.show()
