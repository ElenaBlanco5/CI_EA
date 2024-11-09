import numpy as np
import matplotlib.pyplot as plt
import cma
from tqdm import tqdm
import math

def ackley(x):
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20 + np.e

# Results dictionary
results = {}
for key in ['initial_sol', 'best_sol', 'best_fit', 'iter']:
    results[key] = []

# Set initial parameters
np.random.seed(42)
n = 2
sigma0 = 0.5
sigma0_1 = 0.9
f_name = 'ackley'
options = cma.CMAOptions()
options['popsize'] = 4+math.floor(3*math.log(n))
options['maxiter'] = 200
options['tolfun'] = 1e-10
options['bounds'] = [[-30, -30], [30, 30]]

alg = []
convergence = []
convergence_local = []
convergence_1 = []
convergence_local_1 = []
for i in tqdm(range(50)):
    x01 = np.random.uniform(options['bounds'][0][0], options['bounds'][1][0])
    x02 = np.random.uniform(options['bounds'][0][1], options['bounds'][1][1])
    x0 = (x01,x02)

    # Set up the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    es_1 = cma.CMAEvolutionStrategy(x0, sigma0_1, options)

    # Optimize using CMA-ES
    es.optimize(ackley)
    es_1.optimize(ackley)

    # Get the best solution found
    best_solution = es.result.xbest
    best_value = es.result.fbest
    best_value_1 = es_1.result.fbest
    iter = es.result.iterations
    iter_1 = es_1.result.iterations

    # Store results
    results['initial_sol'].append(x0)
    results['best_sol'].append(best_solution)
    results['best_fit'].append(best_value)
    results['iter'].append(iter)
    alg.append(es)
    if abs(best_value) < 1e-4:
        convergence.append(i)
    elif iter < options['maxiter']:
        convergence_local.append(i)

    if abs(best_value_1) < 1e-4:
        convergence_1.append(i)
    elif iter_1 < options['maxiter']:
        convergence_local_1.append(i)

# Print the results
print(f"Initial pop: {4+math.floor(3*math.log(n))}")
print(f"Best fitness value: {np.min(results['best_fit'])}")
best_index = np.argmin(results['best_fit'])
print(f"Best solution: {results['best_sol'][best_index]}")
print(f"Initial solution: {results['initial_sol'][best_index]}")
print(f"Iterations: {results['iter'][best_index]}")

# Plot initial points with their convergence (sigma=0.5)
fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,3))
conv = [results['initial_sol'][idx] for idx in convergence]
conv_local = [results['initial_sol'][idx] for idx in convergence_local]
no_conv = [el for i,el in enumerate(results['initial_sol']) if i not in convergence+convergence_local]
ax[0].scatter([p[0] for p in conv], [p[1] for p in conv], color='blue', s=10, label='global conv')
ax[0].scatter([p[0] for p in conv_local], [p[1] for p in conv_local], color='orange', s=10, label='local conv')
ax[0].scatter([p[0] for p in no_conv], [p[1] for p in no_conv], color='red', s=10, label='no conv')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_xticks(range(-30,31,10))
ax[0].set_yticks(range(-30,31,10))
ax[0].set_title('Initial points (sigma0=0.5)')
ax[0].set_aspect('equal')
ax[0].legend(loc='upper right', fontsize='small')

# Plot initial points with their convergence (sigma=0.9)
conv_1 = [results['initial_sol'][idx] for idx in convergence_1]
conv_local_1 = [results['initial_sol'][idx] for idx in convergence_local_1]
no_conv_1 = [el for i,el in enumerate(results['initial_sol']) if i not in convergence_1+convergence_local_1]
ax[1].scatter([p[0] for p in conv_1], [p[1] for p in conv_1], color='blue', s=10)
ax[1].scatter([p[0] for p in conv_local_1], [p[1] for p in conv_local_1], color='orange', s=10)
ax[1].scatter([p[0] for p in no_conv_1], [p[1] for p in no_conv_1], color='red', s=10)
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[1].set_xticks(range(-30,31,10))
ax[1].set_yticks(range(-30,31,10))
ax[1].set_title('Initial points (sigma0=0.9)')
ax[1].set_aspect('equal')

# Plot global-local solutions
conv_best_sol = [results['best_sol'][idx] for idx in convergence]
conv_local_best_sol = [results['best_sol'][idx] for idx in convergence_local]
ax[2].scatter([p[0] for p in conv_best_sol], [p[1] for p in conv_best_sol], color='blue', s=10)
ax[2].scatter([p[0] for p in conv_local_best_sol], [p[1] for p in conv_local_best_sol], color='orange', s=10)
ax[2].set_xlabel('x1')
ax[2].set_ylabel('x2')
ax[2].set_aspect('equal')
ax[2].set_title('Convergence points')

# Save figure (3 plots)
fig.savefig(f'plots/{f_name}_points.png', format='png')

# Plot cma results
alg[best_index].logger.plot()
plt.savefig(f'plots/{f_name}_optimization.png', format='png')
