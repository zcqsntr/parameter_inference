import sys
import os
import yaml
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

from hybrid_PS_SGD import *


small_domain = np.array([[470000, 490000],  [0.4, 0.8]])
large_domain = np.array([[100000, 1000000],  [0., 1.]])
domain = small_domain

x = np.linspace(domain[0][0], domain[0][1],100)
y = np.linspace(domain[1][0], domain[1][1],100)
X,Y = np.meshgrid(x,y)



n_points = [i for i in range(2,10)]
n_points += [i for i in range(10, 200, 10)]

for n_point in n_points:

    Z = np.load('Z_' + str(n_point) + '.npy')


    fig = plt.figure(figsize = (12,8))
    im = plt.contour(X,Y,Z, 40, vmin=abs(Z).min(), vmax=abs(Z).max())
    plt.title(str(n_point))

    cb = fig.colorbar(im)
plt.show()
