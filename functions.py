import numpy as np

def salomon(x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    f = 1 - np.cos(2 * np.pi * r) + 0.1 * r
    return f

def hosaki(x):
    return (1 - 8 * x[0] + 7 * x[0]**2 - (7/3) * x[0]**3 + (1/4) * x[0]**4) * x[1]**2 * math.exp(-x[1])

def ackley(x):
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20 + np.e

def trid(x):
    sum1 = sum((x_i - 1)**2 for x_i in x)
    sum2 = sum(x[i] * x[i - 1] for i in range(1, 6))
    return sum1 - sum2