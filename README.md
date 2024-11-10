# Lab-CI: Evolutionary Computation

## Team
- Elena Blanco Lopez
- Clàudia Boixader Garcia

## Options to run the code
There's two python scripts to run directly:
- optimize.py
- test_ackley.py

The first script provides the functionality to execute the CMA-ES algorithm to optimize any function defined in functions.py. The second script, however, is specifically configured for optimizing the Ackley function within a specific setting, which is optimizing the function for the same initial points but changing the sigma value (Section 5, Figure 2). As a result, test_ackley.py has all required parameters pre-set and can be run directly, while optimize.py requires user-specified parameters to define the optimization task for the algorithm.

## Execute optimize.py
When you run the script, the first view that appears is a menu that asks the user to choose between the four functions available to optimize:
```plaintext
Enter the function name you want to optimize:
- Hosaki
- Ackley
- Salomon
- Trid
```

After selecting the function, the following parameters will be required to be entered by the user:

```plaintext
Enter dimension of search space:
```
- Integer > 1

```plaintext
Enter initial sigma:
```
- Float within [0,1]

```plaintext
Enter lower bound for search space (integer):
```
- Integer > 0

```plaintext
Enter upper bound for search space (integer):
```
- Integer > 0
- Integer > Lower bound

## Considerations
Note that, since CMA-ES algorithm samples the population randomly at each generation from a multivariate Gaussian distribution on the current mean of the population, one will not be able to get exactly the same results that are displayed in the pdf document. Moreover, the initial population is also initialized randomly within the search space, but even if we saved the initial generation and runned the algorithm several times, the random component would not disappear because of the selection process. However, the plots generated are all stored inside the ```/plots/``` folder and the parameters that were set to obtain them are listed in the pdf file.
