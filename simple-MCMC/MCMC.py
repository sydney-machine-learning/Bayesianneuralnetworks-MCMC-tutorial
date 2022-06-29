import matplotlib.pyplot as plt
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
    return min(1, ((likelihood(p_new) / likelihood(p)) * (prior(p) / prior(p_new))))


# Define Prior Function - Uniform Distribution
def prior(p):
    return uniform.pdf(p)


# pdf of normal distirbution
def normal_probability(x, mu=0, sigma=1):
    return (np.exp((-((x - mu) ** 2)) / (2 * sigma ** 2))) / (
        sigma * np.sqrt(2 * np.pi)
    )


# main

"""
A simple random-walk Metropolis-Hastings algorithm to simulate a Normal(0, 1) distribution
"""


def main():

    results = []  # Create empty list to store samples

    # Initialzie a value of p from a uniform distribution between -5 and 5
    p = np.random.uniform(-5, 5)

    # Define number of samples and burn-in ratio
    n_samples = 5000
    burn_in = int(0.1 * n_samples)

    count = 0  # count accepted samples
    # Create the MCMC loop
    for i in range(n_samples):

        # Propose a new value of p
        p_new = np.random.uniform(-5, 5)
        # print(i, p_new)

        # Compute acceptance probability
        R = normal_probability(p_new) / normal_probability(p)
        # Draw random sample from a uniform distribution between 0 and 1 to compare to the acceptance probability
        u = np.random.random_sample()

        # If R is greater than u, accept the new value of p (set p = p_new)
        if u < R:
            p = p_new
            count = count + 1

        # Record values
        results.append(p)

    # print(results, "results")
    per_accept = (count / n_samples) * 100
    print(per_accept, "percentage accept")
    pos_mean = np.mean(results)

    # Royce - give histogram and traceplot of results - improve comments and variable names

    print(pos_mean)

    # histogram of the result with only burn-in
    plt.hist(results[burn_in:], bins=20)
    plt.title("Posterior distribution")
    plt.savefig("result_hist.png")
    plt.clf()

    # trace plot of result
    plt.plot(range(burn_in), results[:burn_in], color="red")
    plt.plot(range(burn_in, n_samples), results[burn_in:], color="orange")
    plt.title("Traceplot")
    plt.savefig("trace_plot.png")


if __name__ == "__main__":
    main()
