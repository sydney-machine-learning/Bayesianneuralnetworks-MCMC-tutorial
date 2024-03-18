# Bayesian Neural Networks via MCMC Tutorial

We present a tutorial for MCMC methods that covers simple Bayesian linear and logistic models, and Bayesian neural networks. The aim of this tutorial is to bridge the gap between theory and implementation via coding, given a general sparsity of libraries and tutorials to this end. This tutorial provides code in Python with data and instructions that enable their use and extension. We provide results for some benchmark problems showing the strengths and weaknesses of implementing the respective Bayesian models via MCMC. We highlight the challenges in sampling multi-modal posterior distributions in particular for the case of Bayesian neural networks, and the need for further improvement of  convergence diagnosis.

Code for the Bayesian neural networks via MCMC tutorial:

- `01-Distributions`: Generat and visualise some basic distributions that will be used throughout the tutorial
- `02-Basic-MCMC.ipynb`: Implementing a basic MCMC algorithm
- `03-Linear-Model.ipynb`: Implementing an MCMC algorithm to fit a Bayesian linear regression model
- `04-Bayesian-Neural-Network.ipynb`: Implementing an MCMC algorithm to fit a Bayesian neural network

Additional examples for classification:

- `03a-Linear-Model-Classification.ipynb`: Implementing an MCMC algorithm to fit a Bayesian logistic regression model
- `04a-Bayesian-Neural-Network-Classification.ipynb`: Implementing an MCMC algorithm to fit a Bayesian neural network for classification

Code to reproduce the results in the paper is available in:

- `publication_results/`

## Python code

In the case of Bayesian neural networks, we provide complete code for regression and classiciation case in 'code' folder with sample plots for regression probllem in 'code/py-results' folder. Note that the datasets have been obtained from the following sources: 

- Sunspot dataset for one-step ahead prediction problem: https://www.swpc.noaa.gov/products/solar-cycle-progression 
- Energy dataset for regression problem: https://archive.ics.uci.edu/dataset/242/energy+efficiency
- Ionosphere for binary classification: https://archive.ics.uci.edu/dataset/52/ionosphere
- Iris for multi-class classification:  https://archive.ics.uci.edu/dataset/53/iris

## Docker

A [Docker enviroment for this tutorial is available on DockerHub](https://docs.docker.com/docker-hub/quickstart/) and can be pulled with:

```bash
docker pull jsimdare/mcmc-tutorial:latest

```

## Acknowledgements

Data Analytics for Resources and Environment (DARE), ARC Industrial Transformation Training Centre, University of Sydney, Sydney, Australia (https://darecentre.org.au) and Royce Chen from UNSW Sydney


## Research paper

- R Chandra and J Simmons, Bayesian neural networks via MCMC: a Python-based tutorial, IEEE Access, 2024: [arXiv preprint arXiv:2304.02595](https://arxiv.org/abs/2304.02595)


## Data citations
- Abalone | Ionosphere | Iris: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
- Sunspot: Vanlommel, P., Cugnon, P., Linden, R. V. D., Berghmans, D., & Clette, F. (2004). The SIDC: world data center for the sunspot index. Solar Physics, 224, 113-120.
