import numpy as np
import matplotlib.pyplot as plt
import cma
from tqdm import tqdm
import math
import functions

# Ask for initial parameters
while True:
    print("Enter the function name you want to optimize")
    f_name = input("- Hosaki\n- Ackley\n- Salomon\n- Trid\n")
    f_name = f_name.lower()
    if f_name in ['hosaki','ackley','salomon','trid']: break

while True:
    n = input("Enter dimension of search space:")
    try:
        n = int(n)
        if n > 0: break
    except ValueError:
        continue

while True:
    sigma0 = input("Enter initial sigma:")
    try:
        sigma0 = float(sigma0)
        if (sigma0>=0) and (sigma0<=1): break
    except ValueError:
        continue

while True:
    low_bound = input("Enter lower bound for search space (integer):")
    upp_bound = input("Enter upper bound for search space (integer):")
    try:
        low_bound = int(low_bound)
        upp_bound = int(upp_bound)
        if upp_bound > low_bound: break
    except ValueError:
        continue

# Define fitness value for solution of every function
fit_sol = {'hosaki': -2.3458,
           'ackley': 0,
           'salomon': 0,
           'trid': -50}

# Results dictionary
results = {}
for key in ['initial_sol', 'best_sol', 'best_fit', 'iter']:
    results[key] = []

# Set initial parameters
np.random.seed(42)
options = cma.CMAOptions()
options['popsize'] = 4+math.floor(3*math.log(n))
options['maxiter'] = 200
options['tolfun'] = 1e-10
options['bounds'] = [[low_bound]*n, [upp_bound]*n]
func = getattr(functions, f_name)

alg = []
convergence = []
convergence_local = []
for i in tqdm(range(50)):
    x0 = np.random.uniform(low_bound, upp_bound, n)

    # Set up the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)

    # Optimize using CMA-ES
    es.optimize(func)

    # Get the best solution found
    best_solution = es.result.xbest
    best_value = es.result.fbest
    iter = es.result.iterations

    # Store results
    results['initial_sol'].append(x0)
    results['best_sol'].append(best_solution)
    results['best_fit'].append(best_value)
    results['iter'].append(iter)
    alg.append(es)
    if abs(best_value-(fit_sol[f_name])) < 1e-4:
        convergence.append(i)
    elif iter < options['maxiter']:
        convergence_local.append(i)

# Print the results
print(f"Initial pop: {4+math.floor(3*math.log(n))}")
print(f"Best fitness value: {np.min(results['best_fit'])}")
best_index = np.argmin(results['best_fit'])
print(f"Best solution: {results['best_sol'][best_index]}")
print(f"Initial solution: {results['initial_sol'][best_index]}")
print(f"Iterations: {results['iter'][best_index]}")
print(f"{len(convergence)/50*100}% points reached the global optima\n{len(convergence_local)/50*100} points reached a local optima\n{(50-len(convergence)-len(convergence_local))/50*100} did not converge")

if f_name != 'trid:':
    # Save initial guesses
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
    conv = [results['initial_sol'][idx] for idx in convergence]
    conv_local = [results['initial_sol'][idx] for idx in convergence_local]
    no_conv = [el for i,el in enumerate(results['initial_sol']) if i not in convergence+convergence_local]
    ax[0].scatter([p[0] for p in conv], [p[1] for p in conv], color='blue', s=10, label='global conv')
    ax[0].scatter([p[0] for p in conv_local], [p[1] for p in conv_local], color='orange', s=10, label='local conv')
    ax[0].scatter([p[0] for p in no_conv], [p[1] for p in no_conv], color='red', s=10, label='no conv')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_xticks(range(-20,20,5))
    ax[0].set_yticks(range(-20,20,5))
    ax[0].set_title('Initial points')
    ax[0].set_aspect('equal')
    ax[0].legend(loc='upper right', fontsize='small')

    # Save solution points
    conv_best_sol = [results['best_sol'][idx] for idx in convergence]
    conv_local_best_sol = [results['best_sol'][idx] for idx in convergence_local]
    ax[1].scatter([p[0] for p in conv_best_sol], [p[1] for p in conv_best_sol], color='blue', s=10)
    ax[1].scatter([p[0] for p in conv_local_best_sol], [p[1] for p in conv_local_best_sol], color='orange', s=10)
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_aspect('equal')
    ax[1].set_title('Convergence points')
    fig.savefig(f'plots/{f_name}_points.png', format='png')

# Save cma results
#alg[best_index].logger.plot()
#plt.savefig(f'plots/{f_name}_optimization.png', format='png')
