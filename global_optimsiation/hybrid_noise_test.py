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

from particle_swarm_optimiser import *


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

def eggholder_function(X,Y):
    a = -(Y + 47) * np.sin(np.sqrt(np.abs(X/2 + Y + 47)))
    b = - X*np.sin(np.sqrt(np.abs(X-Y+47)))
    return a + b




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
