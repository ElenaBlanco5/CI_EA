import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma
import math

def salomon(x):
    # Calculate the squared distance from the origin
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)

    # Compute the Salomon function value
    f = 1 - np.cos(2 * np.pi * r) + 0.1 * r
    return f

def hosaki(x):
    return (1 - 8 * x[0] + 7 * x[0]**2 - (7/3) * x[0]**3 + (1/4) * x[0]**4) * x[1]**2 * math.exp(-x[1])

def ackley(x):
    """ Ackley function for optimization problem """
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20 + np.e

def f_step(x):
    # Calculate the sum of squared terms
    return sum((math.floor(xi) + 0.5) ** 2 for xi in x)

# Define the initial guess and the CMA-ES parameters
results = {}
for key in ['initial_sol', 'best_sol', 'best_fit', 'iter']:
    results[key] = []

# Set initial parameters
np.random.seed(42)
n = 2
sigma0 = 0.2
f_name = 'salomon'
options = cma.CMAOptions()
options['popsize'] = 4+math.floor(3*math.log(n))
options['maxiter'] = 100
options['tolfun'] = 1e-13
options['bounds'] = [[-5, -5], [5, 5]]
options['maxfevals'] = 1e3*n**2

alg = []
convergence = []
for i in range(50):
    x01 = np.random.uniform(options['bounds'][0][0], options['bounds'][1][0])
    x02 = np.random.uniform(options['bounds'][0][1], options['bounds'][1][1])
    x0 = (x01,x02)

    # Set up the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)

    # Optimize using CMA-ES
    es.optimize(salomon)

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

    if abs(best_value) < 1e-13:
        convergence.append(i)

# Print the results
print(f"Initial pop: {4+math.floor(3*math.log(n))}")
print(f"Best fitness value: {np.min(results['best_fit'])}")
best_index = np.argmin(results['best_fit'])
print(f"Best solution: {results['best_sol'][best_index]}")
print(f"Initial solution: {results['initial_sol'][best_index]}")
print(f"Iterations: [{np.min(results['iter'])},{np.max(results['iter'])}]")

# Plot initial guesses
x = [p[0] for p in results['initial_sol']]
y = [p[1] for p in results['initial_sol']]
plt.scatter(x[convergence], y[convergence], color='blue', s=10)  # Scatter plot of points
plt.scatter(x[], y[])
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig(f'plots/{f_name}_initial_guesses.png', format='png')

# Plot cma results
alg[best_index].logger.plot()
plt.savefig(f'plots/{f_name}_optimization.png', format='png')
