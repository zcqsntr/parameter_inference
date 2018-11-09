import sympy as sp
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import newton_krylov
from sympy.utilities.lambdify import lambdify, implemented_function



(K_10, K_11, K_20, K_22, mu1_max, mu2_max, a_11, a_12, a_21, a_22, q, y_11, y_22, y_10, y_20) = sp.symbols(
    ['K_10', 'K_11', 'K_20', 'K_22', 'mu1_max', 'mu2_max', 'a_11', 'a_12', 'a_21', 'a_22',
        'q', 'y_11', 'y_22', 'y_10', 'y_20'])

parameters=(K_10, K_11, K_20, K_22, mu1_max, mu2_max, a_11, a_12, a_21, a_22, q, y_11, y_22, y_10, y_20)


# Xs
C1 = sp.Symbol('C1')
C2 = sp.Symbol('C2')
C0 = sp.Symbol('C0')


# Ys
N1 = sp.Symbol('N1')
N2 = sp.Symbol('N2')

x_i = [C1, C2, C0]

y_i = [N1, N2]


# define functions
mu1 = sp.Lambda((C0, C1), mu1_max*C0*C1/((K_10 + C0)*(K_11 + C1)))
mu2 = sp.Lambda((C0, C2), mu2_max*C0*C2/((K_20 + C0)*(K_22 + C2)))


func_dict = {
    'mu1': mu1,
    'mu2': mu2
}

# define differential equations
dN1 = sp.sympify('N1*(mu1(C0, C1) + a_11*N1 + a_12*N2 - q)', locals = func_dict)
dN2 = sp.sympify('N2*(mu2(C0, C2) + a_21*N1 + a_22*N2 - q)', locals = func_dict)
dC1 = sp.sympify('q*(C_1i - C1) - 1/y_11 * mu1(C0, C1) * N1', locals = func_dict)
dC2 = sp.sympify('q*(C_2i - C2) - 1/y_22 * mu2(C0, C2) * N2', locals = func_dict)
dC0 = sp.sympify('q*(C_0i - C0) - 1/y_10*mu1(C0, C1)*N1 - 1/y_20*mu2(C0, C2)*N2', locals = func_dict)


equs = [dN1, dN2, dC1, dC2, dC0]

fi = sp.Matrix(equs)
print(fi)
dN1dp = sp.diff(dN1, mu1_max)


dN1dps = []
for p in parameters:
    dN1dps.append(sp.diff(dN1, p))
print(dN1dps)
