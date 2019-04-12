from gradient_descent_optimiser import *


Q = np.array([1, 0, 0, 1]).reshape(2,2)
R = np.array([0, 0, 0, 0]).reshape(2,2)


GDO = GradientDescentOptimiser(Q, R)
Cins = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/MPC_double_aux_Cins.npy')
full_sol = np.load('/home/neythen/Desktop/Projects/parameter_estimation/system_trajectories/MPC_double_aux.npy')
initial_x = full_sol[0, :2]
all_x = full_sol[:, :2]
initial_params = np.array([-0.2, -0.2, -0.2, -0.2, 2,2])


param_vec = GDO.find_minimum(initial_x, Cins.T, all_x, initial_params, 1000, 10, 'param_est')

np.save('param_vec.npy', param_vec)
