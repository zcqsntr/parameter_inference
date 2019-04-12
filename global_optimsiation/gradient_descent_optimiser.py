import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr

class GradientDescentOptimiser():
    '''
    class for the inference of parameters in linear models.
    '''

    def __init__(self, Q, R):
        self.Q = Q
        self.R = R


    def linear_model(self, x, t, u, A, B):

        if str(type(t)) == '<class \'autograd.numpy.numpy_boxes.ArrayBox\'>': # sort this out
            t = t._value
            t = int(t)
        else:
            t = int(t)
        t = min(u.shape[0] - 1, t) # to prevent solver from going past the max time

        Cin = u[:, t]


        sdot = np.matmul(A, x) + B*Cin*x

        return sdot


    def predict(self, initial_x, u, A, B, time):
        # extract params from param_vec into form used by the rest of the code

        time_points = np.array([x *1 for x in range(time)])

        sol = odeint(self.linear_model, initial_x, time_points, tuple((u, A, B)))[1:]

        pred_x = sol[-1, 0:2]

        return pred_x

    def objective(self, param_vec, current_x, next_x, u, Q, R):
        A = param_vec[:4].reshape(2,2)
        B = param_vec[4:]

        print()

        print(A)
        print(B)
        print()




        x_pred = self.predict(current_x, u, A, B, self.time)

        return np.sum(np.matmul((x_pred - next_x), np.matmul(Q, (x_pred - next_x).T)) + np.matmul(u.T, np.matmul(R, u)))


    def param_est_grad_wrapper(self, param_vec, i):
        '''
        for the autograd optimisers

        param_vec = [flatten(A), flatten(B)]
        '''

        print(i)
        grad_func = grad(self.objective)



        return grad_func(param_vec, self.current_x, self.target_x, self.Cin, self.Q, self.R)

    def MPC_grad_wrapper(self, Cin, i):
        grad_func = grad(self.objective)

        A = param_vec[:4].reshape(2,2)
        B = param_vec[4:].reshape(2,2)

        return grad_func(self.current_x, self.target_x, Cin, A, B, self.Q, self.R)


    def find_minimum(self, current_x, constant, target, initial_guess, n_timesteps, n_steps, mode = 'MPC'):
        self.current_x = current_x
        self.target_x = target
        self.time = n_timesteps
        if mode == 'MPC':
            param_vec = constant
            new_Cin = adam(self.MPC_grad_wrapper, Cin, num_iters = n_steps, step_size = 0.01)
            return new_Cin

        elif mode == 'param_est':
            self.Cin = constant
            param_vec = initial_guess
            new_param_vec = adam(self.param_est_grad_wrapper, param_vec, num_iters = n_steps)
            return new_param_vec
