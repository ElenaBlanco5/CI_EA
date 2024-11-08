import numpy as np
import cma
import math

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

def sphere(x):
    """Sphere function for optimization problem"""
    return np.sum(x ** 2)

def ursem03(x):
    """Ursem03 function for 2D optimization problem"""
    x1, x2 = x[0], x[1]  # Extract x1 and x2 from input array

    # Compute the first term for x1
    term1 = -np.sin(2.2 * np.pi * x1 + 0.5 * np.pi) * ((2 - np.abs(x1)) / 2) * ((3 - np.abs(x1)) / 2)

    # Compute the second term for x2
    term2 = -np.sin(2.2 * np.pi * x2 + 0.5 * np.pi) * ((2 - np.abs(x2)) / 2) * ((3 - np.abs(x2)) / 2)

    # Sum the two terms
    return term1 + term2

# Define the initial guess and the CMA-ES parameters
x0 = np.random.rand(2,1) # Initial guess (10-dimensional)
sigma0 = 0.5  # Initial standard deviation (step size)

options = cma.CMAOptions()
options['popsize'] = 25  # Set population size to 50
options['maxiter'] = 100  # Set maximum iterations to 100
options['tolfun'] = 1e-5  # Set tolerance for function value convergence

# Set up the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(x0, sigma0, options)

# Optimize using CMA-ES
es.optimize(ursem03)  # Provide the fitness function (Ackley function)

# Get the best solution found
best_solution = es.result.xbest
best_value = es.result.fbest

# Print the results
print(f"Best solution: {best_solution}")
print(f"Best fitness value: {best_value}")

