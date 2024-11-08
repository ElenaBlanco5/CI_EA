import numpy as np
import cma

def ackley(x):
    """ Ackley function for optimization problem """
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20 + np.e

# Define the initial guess and the CMA-ES parameters
x0 = np.random.rand(10)  # Initial guess (10-dimensional)
sigma0 = 0.5  # Initial standard deviation (step size)

# Set up the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(x0, sigma0)

# Optimize using CMA-ES
es.optimize(ackley)  # Provide the fitness function (Ackley function)

# Get the best solution found
best_solution = es.result.xbest
best_value = es.result.fbest

# Print the results
print(f"Best solution: {best_solution}")
print(f"Best fitness value: {best_value}")

