import numpy as np

minima = np.array(np.load('minima.npy'))
parameters = np.array(np.load('parameters.npy'))
print(minima.shape)
print(parameters.shape)


true_minimas = np.array([-1.031628453489877, 0, -2.062611870822739])
minima_SSEs = []


for i, function_minima in enumerate(minima): # minima found for each function
    SSE = 0
    for j in range(len(function_minima)):

        true_min = true_minimas[i]
        found_min = min(function_minima[j]) # get minimum over all groups
        SSE += (true_min - found_min)**2
        print(i, (true_min - found_min)**2)
    minima_SSEs.append(SSE)

print(minima_SSEs)



true_parameters = np.array([[[0.08984201368301331,-0.7126564032704135], [-0.08984201368301331, 0.7126564032704135]], [[0,0]],[[1.349406608602084, 1.349406608602084], [-1.349406608602084, 1.349406608602084],[1.349406608602084,-1.349406608602084], [-1.349406608602084, -1.349406608602084]]])
for i, function_parameters in enumerate(parameters):
    true_params = true_parameters[i]
    n_failed = 0
    for params in function_parameters:
        # check that at least one of params is close to each of the true minima for this function
        for tp in true_params: # probably the worst code ive ever written
            found = False
            for p in params:
                if np.linalg.norm(tp-p) < 0.0000001:
                    found = True
            if not found:
                n_failed += 1
    print(n_failed)
