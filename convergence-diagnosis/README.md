# Convergence

## Gelman-Rubin Diagnosis

The Gelman-Rubin (GR) diagnostic evaluates MCMC convergence by analyzing the behaviour of multiple Markov chains. Given multiple chains from different experimental runs, assessment is done by comparing the estimated between-chains and within-chain variances for each parameter, where large differences between the variances indicate non-convergence.

We calculate the potential scale reduction factor (PSRF) which gives the ratio of the current variance in the posterior variance for each parameter compared to that being sampled. The values for the PSRF near 1 indicates convergence.

The following variables are calculated. The between-chain variance is given by
$$ B = \frac{n}{m-1} \sum_{j=1}^m (\bar{\theta_{j}} - \bar{\bar{\theta}})^2 ,$$

and the within-chain variance is given by
$$ W = \frac{1}{m} \sum_{j=1}^m s_j^2 ,$$

where $n$ is the number of samples, $m$ is the number of chains, $ \bar{\theta_{j}} = \frac{1}{n} \sum_{i=1}^n \theta_{ij} $ , $ \bar{\bar{\theta}}  = \frac{1}{m} \sum_{j=1}^m \bar{\theta_j} $, and $s_j^2 = \frac{1}{n-1} \sum_{j=1}^n \theta_{ij} - \bar{\theta_j}$.

After observing these estimates, we may then estimate the target variance $\sigma^2$ with $\hat{\sigma}^2 = \frac{n-1}{n}W + \frac{B}{n}$. Then what is known about $\theta$ can be estimated and the result is an approximate Student t's distribution for $\theta$ with centre $\bar{\bar{\theta}}$, scale $\sqrt{\hat{V}} = \sqrt{\hat{\sigma}^2 + \frac{B}{mn}}$ and degrees of freedom $d = \frac{2\hat{V}}{\hat{\text{var}} (\hat{V})}$. Here 

$$
\begin{align*}
\hat{\text{var}} (\hat{V}) &= \left( \frac{n-1}{n} \right)^2 \frac{1}{m} \hat{\text{var}}(s_i^2) + \left( \frac{m+1}{mn} \right)^2 \frac{2}{m-1} B^2 \\
						   &+ 2\frac{(m+1)(n-1)}{mn^2} \\
						   &\cdot \frac{n}{m} [\hat{\text{cov}}(s_i^2, \bar{\theta_{i}^2}) - 2\bar{\bar{\theta}}\hat{\text{cov}}(s_i^2, \bar{\theta_i})] .
\end{align*}
$$

Finally, we calculate the potential scale reduction factor (PSRF)
$$ \sqrt{\hat{R}} = \sqrt{\frac{\hat{V}}{W} \frac{d}{d-2}} .$$