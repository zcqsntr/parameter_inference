import sys
import os
import yaml

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import cm
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

from hybrid_PS_SGD_v2 import *


def six_hump_camel(parameters, *args):

    '''
    test function for optimisation algorithms
    domain:
        X[0] = [-3,3], X[1] = [-2,2]
    minima:
        f(X*) = −1.031628453489877 at X* = +-[0.08984201368301331, -0.7126564032704135]
    '''
    X,Y = parameters
    return (4 - 2.1*X**2 + X**4/3)*X**2 + X*Y + (-4 + 4*Y**2)*Y**2

def ackleys_function(parameters, *args):
    '''
    domian x1,x2 = [-40,40]
    minima of 0 at [0,0]
    '''
    X,Y = parameters
    firstSum = X**2.0 + Y**2.0
    secondSum = np.cos(2.0*math.pi*X) + np.cos(2.0*math.pi*Y)
    return -20.0*np.exp(-0.2*np.sqrt(firstSum/2)) - np.exp(secondSum/2) + 20 + np.e

def eggholder_function(parameters, *args):
    '''
    domain:  -512 < x,y < 512
    minima: f(512,404.2319)=-959.6407
    '''
    X,Y = parameters
    a = -(Y + 47) * np.sin(np.sqrt(np.abs(X/2 + Y + 47)))
    b = - X*np.sin(np.sqrt(np.abs(X-Y+47)))
    return a + b

def crossintray_function(parameters, *args):
    '''
    f(+-1.349406608602084, +- 1.349406608602084) = −2.062611870822739 (four minima)
    '''
    X,Y = parameters
    return -0.0001*(np.abs(np.sin(X)*np.sin(Y)*np.exp(np.abs(100-np.sqrt(X**2 + Y**2)/np.pi)))+1)**0.1


domains = [np.array([[-3, 3],[-2,2]]), np.array([[-40, 40], [-40, 40]]), np.array([[-10, 10], [-10, 10]])]
functions = [six_hump_camel, ackleys_function, crossintray_function]
n_groups = [5,5,10]
minima = [[],[],[]]
parameters = [[],[],[]]

n_particles = 50
num_repeats = 100
if __name__ == '__main__':

    for i, (function, domain, n_group) in enumerate(zip(functions, domains, n_groups)):
        for _ in range(num_repeats):

            '''
            fig = plt.figure(figsize = (16,12))
            im = plt.contour(X,Y,Z, vmin=abs(Z).min(), vmax=abs(Z).max())

            cb = fig.colorbar(im)
            ims = []
            '''
            swarm = Swarm(domain, n_particles, n_group, [2,2], loss_function = function)
            values, positions, ims = swarm.find_minimum(None, None, None, 1000, None)
            print(values, positions)
            minima[i].append(values)
            parameters[i].append(positions)

            '''
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100000)
            ani.save('hybrid.mp4', bitrate = 1000)
            '''

            '''
            for p in swarm.particles:
                im = plt.plot(*p.position, 'ro')
            '''

    minima = np.array(minima)
    parameters = np.array(parameters)
    np.save('minima.npy', minima)
    np.save('parameters.npy', parameters)
    #plt.show()

    # sample particles from region X[0] = [-3,3], X[1] = [-2,2]
