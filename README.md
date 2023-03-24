# Bayesian Neural Networks via MCMC Tutorial

We present a tutorial for MCMC methods that covers simple Bayesian linear and logistic models, and Bayesian neural networks. The aim of this tutorial is to bridge the gap between theory and implementation via coding, given a general sparsity of libraries and tutorials to this end. This tutorial provides code in Python with data and instructions that enable their use and extension. We provide results for some benchmark problems showing the strengths and weaknesses of implementing the respective Bayesian models via MCMC. We highlight the challenges in sampling multi-modal posterior distributions in particular for the case of Bayesian neural networks, and the need for further improvement of  convergence diagnosis.

Code for the Bayesian neural networks via MCMC tutorial:

- `01 Distributions`: Generat and visualise some basic distributions that will be used throughout the tutorial
- `02 Basic MCMC.ipynb`: Implementing a basic MCMC algorithm
- `03 Linear Model.ipynb`: Implementing an MCMC algorithm to fit a Bayesian linear regression model
- `04 Bayesian Neural Network.ipynb`: Implementing an MCMC algorithm to fit a Bayesian neural network

Code to reproduce the results in the paper is available in:

- `publication_results/`

## Acknowledgements

Data Analytics for Resources and Environment (DARE), ARC Industrial Transformation Training Centre, University of Sydney, Sydney, Australia (https://darecentre.org.au)


## Data citations
- Abalone | Ionosphere | Iris: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
- Sunspot: Vanlommel, P., Cugnon, P., Linden, R. V. D., Berghmans, D., & Clette, F. (2004). The SIDC: world data center for the sunspot index. Solar Physics, 224, 113-120.