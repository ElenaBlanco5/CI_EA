import math
import numpy as np

def f_step(x):
    # Calculate the sum of squared terms
    return sum((math.floor(xi) + 0.5) ** 2 for xi in x)

def felli(x):
    N = x.shape[0]
    if N < 2:
        raise ValueError("Dimension must be greater than one.")
    return np.sum(1e6 ** (np.arange(N) / (N - 1)) * x**2)

# User defined input parameters (need to be edited)
N = 10  # number of objective variables/problem dimension
xmean = np.random.rand(N, 1)  # objective variables initial point
sigma = 0.5  # coordinate-wise standard deviation (step-size)
stopfitness = 1e-10  # stop if fitness < stopfitness (minimization)
stopeval = 1e3 * N**2  # stop after stopeval number of function evaluations

# Strategy parameter setting: Selection
lambda_ = 4 + int(3 * np.log(N))  # population size, offspring number
mu = lambda_ // 2  # number of parents/points for recombination
weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))  # recombination weights
weights /= np.sum(weights)  # normalize recombination weights array
mueff = np.sum(weights)**2 / np.sum(weights**2)  # variance-effective size of mu

# Strategy parameter setting: Adaptation
cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C
cs = (mueff + 2) / (N + mueff + 5)  # time constant for cumulation for sigma control
c1 = 2 / ((N + 1.3)**2 + mueff)  # learning rate for rank-one update of C
cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + 2 * mueff / 2))  # for rank-mu update
damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs  # damping for sigma

# Initialize dynamic (internal) strategy parameters and constants
pc = np.zeros((1,N))  # evolution path for C
ps = np.zeros((1,N))  # evolution path for sigma
B = np.eye(N)  # B defines the coordinate system
D = np.eye(N)  # diagonal matrix D defines the scaling
C = B @ D @ (B @ D).T  # covariance matrix
eigeneval = 0  # B and D updated at counteval == 0
chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))  # expectation of ||N(0,I)|| == norm(randn(N,1))

# Generation Loop
counteval = 0  # initialize evaluation counter

# Pre-allocate arrays for offspring and fitness
arz = np.zeros((lambda_, N))
arx = np.zeros((lambda_, N))
arfitness = np.zeros(lambda_)

while counteval < stopeval:
    # Generate and evaluate lambda offspring
    for k in range(lambda_):
        arz[:, k] = np.random.randn(N)  # standard normally distributed vector
        arx[:, k] = xmean.flatten() + sigma * (B @ D @ arz[:, k])  # add mutation, Eq. 40
        # Evaluate the fitness function (replace 'felli' with your actual fitness function)
        arfitness[k] = felli(arx[:,k])#f_step(arx[:, k])  # objective function call
        counteval += 1

    # Sort by fitness and compute weighted mean into xmean
    arindex = np.argsort(arfitness)  # indices of sorted fitness (for minimization)
    xmean = arx[:, arindex[:mu]] @ weights  # recombination, Eq. 42
    zmean = arz[:, arindex[:mu]] @ weights  # == D^(-1)*B’*(xmean - xold) / sigma

    # Cumulation: Update evolution paths
    ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (B @ zmean)  # Eq. 43
    hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) / chiN
            < 1.4 + 2 / (N + 1))
    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ D @ zmean)

    # Adapt covariance matrix C
    C = ((1 - c1 - cmu) * C  # regard old matrix, Eq. 47
         + c1 * ((pc @ pc.T)  # plus rank one update
                 + (1 - hsig) * cc * (2 - cc) * C)  # minor correction
         + cmu * (B @ D @ arz[:, arindex[:mu]])  # plus rank mu update
         @ np.diag(weights) @ (B @ D @ arz[:, arindex[:mu]]).T)

    # Adapt step-size sigma
    sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))  # Eq. 44

    # Update B and D from C
    if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:  # to achieve O(N^2) complexity
        eigeneval = counteval
        C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
        D, B = np.linalg.eigh(C)  # eigen decomposition; B = eigenvectors, D = eigenvalues
        D = np.diag(np.sqrt(D))  # D now contains the standard deviations

    # Break if the fitness is good enough
    if arfitness[arindex[0]] <= stopfitness:
        print("Optimization terminated: stopfitness reached.")
        break

    # Escape flat fitness, or issue a warning if the fitness does not improve
    if arfitness[0] == arfitness[int(0.7 * lambda_)]:
        sigma *= np.exp(0.2 + cs / damps)
        print("Warning: Flat fitness landscape detected. Consider reformulating the objective function.")

    # Display progress
    print(f"{counteval}: Best fitness = {arfitness[arindex[0]]}, Minimum = {arindex[0]}")

# Final Message
print(f"{counteval}: Best fitness = {arfitness[arindex[0]]}, Minimum = {arindex[0]}")
xmin = arx[:, arindex[0]]  # Return best point of last generation.
# Notice that xmean is expected to be even better.



