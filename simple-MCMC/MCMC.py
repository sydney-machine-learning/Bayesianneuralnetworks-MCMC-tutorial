
import numpy as np
from scipy.stats import binom, uniform


# Define the Likelihood P(x|p) - binomial distribution
def likelihood(p):
    # Define  data
    x = 100
    n = 100
# @Sahani - can be update this so this is a difference between two similar distributions so that it gives a better analogy of predicted vs actual?

    return binom.pmf(x, n, p)
    
# This function will accept the current and proposed values 
def acceptance_ratio(p, p_new):
    # Return R, using the functions we created before
    return min(1, ((likelihood(p_new) / likelihood(p)) * (prior(p_new) / prior(p))))
    
# Define Prior Function - Uniform Distribution    
def prior(p):
    return uniform.pdf(p)

#main 
    
results = [] # Create empty list to store samples
    
# Initialzie a value of p
p = np.random.uniform(0, 1)

# Define model parameters
n_samples = 500
burn_in = 50
 
count = 0 #count accepted samples
# Create the MCMC loop
for i in range(n_samples):
    # Propose a new value of p randomly from a uniform distribution between 0 and 1
    p_new = np.random.random_sample()
    print(i, p_new)
    # Compute acceptance probability
    R = acceptance_ratio(p, p_new)
    # Draw random sample to compare R to
    u = np.random.random_sample()
    # If R is greater than u, accept the new value of p (set p = p_new)
    if u < R:
        p = p_new
        count = count +1
    # Record values after burn in 
    if i > burn_in:
        results.append(p)
print(results, 'results')
per_accept = (count/n_samples)*100
print(per_accept, 'percentage accept')
pos_mean = np.mean(results)

# Royce - give histogram and traceplot of results - improve comments and variable names

print(pos_mean)
