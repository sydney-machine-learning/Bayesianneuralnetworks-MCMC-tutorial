# %% [markdown]
# # Bayesian Neural Network MCMC
# Part of the Bayesian neural networks via MCMC: a Python-based tutorial
# 
# This section of the tutorial covers the development of an MCMC algorithm applied to a Neural Network.

# %% [markdown]
# ### Imports

# %%
%load_ext autoreload
%autoreload 2
%pdb 1

# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import binom, uniform
from ipywidgets import interact, fixed, widgets
from tqdm import tqdm
# visulisation function
from functions.visualisations import (
    histogram_trace, plot_y_timeseries, 
    plot_ycorr_scatter, boxplot_weights,
    plot_linear_data
)

from types import MethodType

np.random.seed(2023)

# %%
os.chdir('/project')

# %% [markdown]
# ## Define a class with the functions and attributes required for a neural network model
# First lets define our forward and backward pass functions which form the core of our model. These will be integrated into the NeuralNetwork class and thus contain references to class attributes.
# 
# - `forward_pass`: Function to calculate the output of the network from the input data ($x$)
#     - for our hidden layer - $h_{output} = g\left(x \cdot w_{h} + \delta_{h} \right)$ 
#     - subsequently for the output layer - $f\left(x\right) = o_{output} = g\left(h_{output} \cdot w_{o} + \delta_{o} \right)$
#     - where $g(\cdot)$ is the sigmoid activation function, $w_{h}$ and $w_{o}$ are the weights for the hidden and output layers, and $\delta_{h}$ and $\delta_{o}$ are the biases for the hidden and output layers, respectively.
# - `backward_pass`: Function to update the model parameters using Langevin dynamics
#     - $\bar{\theta_p} = \theta_p + r \times \nabla E_{y_{\mathcal{A}_{D,T}}}\left[\theta_p\right]$
#     - $\nabla E_{y_{\mathcal{A}_{D,T}}}\left[\theta_p\right] = \left(\frac{\delta E}{\delta \theta_1},\dots, \frac{\delta E}{\delta \theta_L} \right)$
#     - $E_{y_{\mathcal{A}_{D,T}}}\left[\theta_p\right] = \sum_{t \in \mathcal{A}_{D,T}} \left(y_t - f\left(x_t\right)^{[k]}\right)^2$
#     - where $\theta = \left(w_{h},w_{o},\delta_{h},\delta_{o}\right)$ comprises the weights and biases of the NN model

# %%
# NN prediction
def forward_pass(self, X):
    '''
    Take an input X and return the output of the network
    Input:
        - X: (N x num_features) array of input data
    Output:
        - self.l2_output: (N) array of output data f(x) which can be 
        compared to observations (Y)
    '''
    # Hidden layer
    l1_z = np.dot(X, self.l1_weights) + self.l1_biases
    self.l1_output = self.sigmoid(l1_z) # activation function g(.)
    # Output layer
    l2_z = np.dot(self.l1_output, self.l2_weights) + self.l2_biases
    self.l2_output = self.sigmoid(l2_z)
    return self.l2_output

def backward_pass(self, X, Y):
    '''
    Compute the gradients using a backward pass and undertake Langevin-gradient 
    updating of parameters
    Input:
        - X: (N x num_features) array of input data
        - Y: (N) array of target data
    '''
    if self.data_case == 'classification':
        # then we need to take the Y values and transform to one-hot encoding
        Y_transformed = np.zeros((self.output_num,))
        Y_transformed[Y.astype(int)] = 1
    else:
        # if regression, then we don't need to transform
        Y_transformed = Y
    # dE/dtheta
    l2_delta = (Y_transformed - self.l2_output) * (self.l2_output * (1 - self.l2_output))
    l2_weights_delta = np.outer(
        self.l1_output,
        l2_delta
    )
    # backprop of l2_delta and same as above
    l1_delta = np.dot(l2_delta,self.l2_weights.T) * (self.l1_output * (1 - self.l1_output))        
    l1_weights_delta = np.outer(
        X,
        l1_delta
    )

    # update for output layer
    self.l2_weights += self.lrate * l2_weights_delta
    self.l2_biases += self.lrate * l2_delta
    # update for hidden layer
    self.l1_weights += self.lrate * l1_weights_delta
    self.l1_biases += self.lrate * l1_delta


# %% [markdown]
# Below we the NeuralNetwork class, which constitutes our model. For readability we have defined the forward and backward passes above, but we will add these to the NeuralNetwork class below. The class contains the following functions:
# 
# - `initialise_network`: Function to initialise the weights and biases of the network in both the hidden and output layers
# - `evaluate_proposal`: Function to assign a proposed $\theta$ to the weights and biases and forward_pass the input data
# - `langevin_gradient`: Function to calculate the Langevin gradient based parameter updates. This uses the gradient computation in the `backward_pass` detailed above
# - `sigmoid`: Sigmoid activation function
# - `encode`: Return a vector $\theta$ of the current weights and biases of the network
# - `decode`: Take a vector $\theta$ and assign it to the weights and biases of the network
# 

# %%
class NeuralNetwork:
    '''
    Neural Network model with a single hidden layer and a single output (y)
    '''
    def __init__(self, layer_sizes,learning_rate=0.01, data_case='regression'):
        '''
        Initialize the model
        Input:
            - layer_sizes (input, hidden, output): array specifying the number of 
            nodes in each layer
            - learning_rate: learning rate for the gradient update
        '''
        # Initial values of model parameters
        self.input_num = layer_sizes[0]
        self.hidden_num = layer_sizes[1]
        self.output_num = layer_sizes[2]

        # total number of parameters from weights and biases
        self.n_params = (self.input_num * self.hidden_num) + (self.hidden_num * self.output_num) +\
            self.hidden_num + self.output_num
        # learning params
        self.lrate = learning_rate

        # Initialize network structure
        self.initialise_network()

        # functions defined above - this is poor practice, but done for readability 
        # and clarity
        self.forward_pass = MethodType(forward_pass, self)
        self.backward_pass = MethodType(backward_pass, self)
        self.data_case = data_case

    def initialise_network(self):
        '''
        Initialize network structure - weights and biases for the hidden layer
        and output layer
        '''
        # hidden layer
        self.l1_weights = np.random.normal(
            loc=0, scale=1/np.sqrt(self.input_num),
            size=(self.input_num, self.hidden_num))
        self.l1_biases = np.random.normal(
            loc=0, scale=1/np.sqrt(self.hidden_num), 
            size=(self.hidden_num,))
        # placeholder for storing the hidden layer values
        self.l1_output = np.zeros((1, self.hidden_num))

        # output layer
        self.l2_weights = np.random.normal(
            loc=0, scale=1/np.sqrt(self.hidden_num), 
            size=(self.hidden_num, self.output_num))
        self.l2_biases = np.random.normal(
            loc=0, scale=1/np.sqrt(self.hidden_num), 
            size=(self.output_num,))
        # placeholder for storing the model outputs
        self.l2_output = np.zeros((1, self.output_num))

    def evaluate_proposal(self, x_data, theta):
        '''
        A helper function to take the input data and proposed parameter sample 
        and return the prediction
        Input:
            data: (N x num_features) array of data
            theta: (w,v,b_h,b_o) vector of parameters with weights and biases
        '''
        self.decode(theta)  # method to decode w into W1, W2, B1, B2.
        size = x_data.shape[0]

        fx = np.zeros(size)
        prob = np.zeros((size,self.output_num))

        for i in range(0, size):  # to see what fx is produced by your current weight update
            fx_tmp = self.forward_pass(x_data[i,])
            if self.data_case == 'classification':
                fx[i] = np.argmax(fx_tmp)
                prob[i,:] = self.softmax(fx_tmp)
            else:
                # regression
                fx[i] = fx_tmp

        return fx, prob

    def langevin_gradient(self, x_data, y_data, theta, depth):
        '''
        Compute the Langevin gradient based proposal distribution
        Input:
            - x_data: (N x num_features) array of input data
            - y_data: (N) array of target data
            - theta: (w,v,b_h,b_o) vector of proposed parameters.
            - depth: SGD depth
        Output: 
            - theta_updated: Updated parameter proposal

        '''
        self.decode(theta)  # method to decode w into W1, W2, B1, B2.
        size = x_data.shape[0] 
        # Update the parameters based on LG 
        for _ in range(0, depth):
            for ii in range(0, size):
                self.forward_pass(x_data[ii,])
                self.backward_pass(x_data[ii,], y_data[ii])
        theta_updated = self.encode()
        return  theta_updated

    # Helper functions
    def sigmoid(self, x):
        '''
        Implentation of the sigmoid function
        '''
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        '''
        Implentation of the softmax function
        '''
        prob = np.exp(x) / np.sum(np.exp(x))
        return prob
        
    def encode(self):
        '''
        Encode the model parameters into a vector
        Output:
            - theta: vector of parameters.
        '''
        w1 = self.l1_weights.ravel()
        w2 = self.l2_weights.ravel()
        theta = np.concatenate([w1, w2, self.l1_biases, self.l2_biases])
        return theta
        
    def decode(self, theta):
        '''
        Decode the model parameters from a vector
        Input:
            - theta: vector of parameters.
        '''
        w_layer1size = self.input_num * self.hidden_num
        w_layer2size = self.hidden_num * self.output_num

        w_layer1 = theta[0:w_layer1size]
        self.l1_weights = np.reshape(w_layer1, (self.input_num, self.hidden_num))

        w_layer2 = theta[w_layer1size:w_layer1size + w_layer2size]
        self.l2_weights = np.reshape(w_layer2, (self.hidden_num, self.output_num))
        self.l1_biases = theta[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.hidden_num]
        self.l2_biases = theta[w_layer1size + w_layer2size + self.hidden_num:w_layer1size + w_layer2size + self.hidden_num + self.output_num]

# %% [markdown]
# ## Define a class for the MCMC sampling
# First we will define our `likelihood_function` and `prior_likelihood` functions. These will be integrated into the MCMC class and thus contain references to class attributes.
# 
# 
# - `prior_likelihood`: Function to calculate the prior likelihood
#     - We assume a Gaussian prior distribution - see Equation 22
#     - $p(\boldsymbol{\theta}) \propto \frac{1}{(2\pi\sigma^2)^{L/2}}\times 
#  \exp\Bigg\{-\frac{1}{2\sigma^2}\bigg( \sum_{i=1}^M \theta^2_i \bigg) \Bigg\}
#  \times \tau^{-2(1+\nu_1)}\exp\left(\frac{-\nu_2}{\tau^2}\right)$
#     - we implement this (using log laws) as:
#     - $\log\left(p\left(\theta\right)\right) = -\frac{L}{2}\log\left(\sigma^2\right) - \frac{1}{2\sigma^2}\sum_{i=1}^M \theta^2_i - \left(1 + \nu_1\right)\log\left(\tau^2\right) - \frac{\nu_2}{\tau^2}$
#     - XX we lose the pi here but not in likelihood_function - consistency
#     
# - `likelihood_function`: Function to calculate the likelihood of the data given the current weights and biases of the network
#     - Our likelihood following Equation 15 is:
#     - $p({\mathbf Y}|X,\boldsymbol{\theta}) = \frac{1}{(2\pi \tau ^2)^{S/2}}\times \exp\left( -\frac{1}{2\tau^2}\sum_{t=1}^S \left( y_{t}- f(\mathbf x_t, \theta)\right)^2\right)$
#     - $\log{\left(p({\mathbf Y}|X,\boldsymbol{\theta})\right)} = -\frac{1}{2}\log{\left(2\pi\tau^2\right)} - \frac{1}{2\tau^2}\sum_{t=1}^S \left(y_{t}- f(\mathbf x_t, \theta)\right)^2$
#     - XX we lose S here

# %%
# Define the likelihood function
def likelihood_function(self, theta, tausq, test=False):
    '''
    Calculate the likelihood of the data given the parameters
    Input:
        theta: (M + 1) vector of parameters. The last element of theta consitutes the bias term (giving M + 1 elements)
        tausq: variance of the error term
    Output:
        log_likelihood: log likelihood of the data given the parameters
        model_prediction: prediction of the model given the parameters
        accuracy: accuracy (RMSE) of the model given the parameters
    '''
    # first make a prediction with parameters theta
    if test:
        x_data = self.x_test
        y_data = self.y_test
    else:
        x_data = self.x_data
        y_data = self.y_data
    model_prediction, _ = self.model.evaluate_proposal(x_data, theta)
    model_simulation = model_prediction + np.random.normal(0,tausq,size=model_prediction.shape) 
    accuracy = self.rmse(model_prediction, y_data) #RMSE error metric 
    # now calculate the log likelihood
    log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(y_data - model_prediction) / tausq)
    return [log_likelihood, model_prediction, model_simulation, accuracy] 

# Define the prior
def prior_likelihood(self, sigma_squared, nu_1, nu_2, theta, tausq): 
    '''
    Calculate the prior likelihood of the parameters
    Input:
        sigma_squared: variance of normal prior for theta
        nu_1: parameter nu_1 of the inverse gamma prior for tau^2
        nu_2: parameter nu_2 of the inverse gamma prior for tau^2
        theta: (M + 1) vector of parameters. The last element of theta consitutes the bias term (giving M + 1 elements)
        tausq: variance of the error term
    Output:
        log_prior: log prior likelihood
    '''
    n_params = self.theta_size # number of parameters in model
    part1 = -1 * (n_params / 2) * np.log(sigma_squared)
    part2 = 1 / (2 * sigma_squared) * (sum(np.square(theta)))
    log_prior = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
    return log_prior

# %%
# For classification problems we will need an adjusted log likelihood and prior
# Define the likelihood function
def classification_likelihood_function(self, theta, tausq=None, test=False):
    '''
    Calculate the likelihood of the data given the parameters
    Input:
        theta: (M + 1) vector of parameters. The last element of theta consitutes the bias term (giving M + 1 elements)
        tausq: variance of the error term
    Output:
        log_likelihood: log likelihood of the data given the parameters
        model_prediction: prediction of the model given the parameters
        accuracy: accuracy (RMSE) of the model given the parameters
    '''
    # first make a prediction with parameters theta
    if test:
        x_data = self.x_test
        y_data = self.y_test
    else:
        x_data = self.x_data
        y_data = self.y_data
    model_prediction, probs = self.model.evaluate_proposal(x_data, theta)
    model_simulation = model_prediction # tausq unused for classification
    accuracy = self.accuracy(model_prediction, y_data) #Accuracy error metric 
    # now calculate the log likelihood
    log_likelihood = 0
    for ii in np.arange(x_data.shape[0]):
        for jj in np.arange(self.model.output_num):
            if y_data[ii] == jj:
                log_likelihood += np.log(probs[ii,jj])    
    return [log_likelihood, model_prediction, model_simulation, accuracy] 

# Define the prior
def classification_prior_likelihood(self, sigma_squared, nu_1, nu_2, theta, tausq=None): 
    '''
    Calculate the prior likelihood of the parameters
    Input:
        sigma_squared: variance of normal prior for theta
        nu_1: parameter nu_1 of the inverse gamma prior for tau^2
        nu_2: parameter nu_2 of the inverse gamma prior for tau^2
        theta: (M + 1) vector of parameters. The last element of theta consitutes the bias term (giving M + 1 elements)
        tausq: variance of the error term
    Output:
        log_prior: log prior likelihood
    '''
    n_params = self.theta_size # number of parameters in model
    part1 = -1 * (n_params / 2) * np.log(sigma_squared)
    part2 = 1 / (2 * sigma_squared) * (sum(np.square(theta)))
    log_prior = part1 - part2
    return log_prior

# %% [markdown]
# Next we define the MCMC sampler as outlined in the text:
# 
# <img src="img/LG_MCMC_algorithm.png" width="50%">
# 
# - The acceptance probability is
#     - $\alpha = \Bigg(1, \frac{p\left(\theta^p_s|Y\right)q\left(\theta^{[i]}|\theta^{p}\right)}{p\left(\theta^{[i]}|Y\right)q\left(\theta^{p}|\theta^{[i]}\right)}\Bigg)$
# - `diff_prop` gives the q ratio
#     - $\frac{q\left(\theta^{[i]}|\theta^{p}\right)}{q\left(\theta^{p}|\theta^{[i]}\right)}$ XX from Equation 24
# - the log posterior ratio:
#     - $\frac{p\left(\theta^p_s|Y\right)}{p\left(\theta^{[i]}|Y\right)}$
#     - where the log posterior is given by the sum of the prior and likelihood:
#     - $\log\left(p\left(\theta|Y\right)\right) = \log\left(p\left(\theta\right)\right) + \log\left(p\left(Y|X,\theta\right)\right)$
#     - is calculated from the following parts:
#     - `diff_likelihood`: gives the likelihood - $\log\left(p\left(Y|X,\theta\right)\right)$ - component of log posterior ratio  
#     - `diff_priorlikelihood` gives the prior - $\log\left(p\left(\theta\right)\right)$ - component of log posterior ratio  

# %%
# MCMC sampler
def sampler(self):
    '''
    Run the sampler for a defined Neural Network model
    '''
    # define empty arrays to store the sampled posterior values
    # posterior of all weights and bias over all samples
    pos_theta = np.ones((self.n_samples, self.theta_size)) 
    # posterior defining the variance of the noise in predictions
    pos_tau = np.ones((self.n_samples, 1))
    pos_eta = np.ones((self.n_samples, 1))

    # record output f(x) over all samples
    pred_y = np.zeros((self.n_samples, self.x_data.shape[0]))
    # record simulated values f(x) + error over all samples 
    sim_y = np.zeros((self.n_samples, self.x_data.shape[0]))
    # record the RMSE of each sample
    rmse_data = np.zeros(self.n_samples)
    # now for test
    test_pred_y = np.ones((self.n_samples, self.x_test.shape[0]))
    test_sim_y = np.ones((self.n_samples, self.x_test.shape[0]))
    test_rmse_data = np.zeros(self.n_samples)

    ## Initialisation
    # initialise theta
    theta = np.random.randn(self.theta_size)
    # make initial prediction
    pred_y[0,], _ = self.model.evaluate_proposal(self.x_data, theta)

    # initialise eta
    eta = np.log(np.var(pred_y[0,] - self.y_data))
    tau_proposal = np.exp(eta)

    # Hyperpriors - considered by looking at distribution of  similar trained  models - i.e distribution of weights and bias
    sigma_squared = self.sigma_squared
    nu_1 = self.nu_1
    nu_2 = self.nu_2

    # calculate the prior likelihood
    prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, theta, tau_proposal)
    # calculate the likelihood considering observations
    [likelihood, pred_y[0,], sim_y[0,], rmse_data[0]] = self.likelihood_function(theta, tau_proposal)

    n_accept = 0  
    n_langevin = 0
    # Run the MCMC sample for n_samples
    for ii in tqdm(np.arange(1,self.n_samples)):
        # Sample new values for theta and tau
        theta_proposal = theta + np.random.normal(0, self.step_theta, self.theta_size)

        lx = np.random.uniform(0,1,1)
        if (self.use_langevin_gradients is True) and (lx < self.l_prob):  
            theta_gd = self.model.langevin_gradient(self.x_data, self.y_data, theta.copy(), self.sgd_depth)  
            theta_proposal = np.random.normal(theta_gd, self.step_theta, self.theta_size)
            theta_proposal_gd = self.model.langevin_gradient(self.x_data, self.y_data, theta_proposal.copy(), self.sgd_depth) 

            # for numerical reasons, we will provide a simplified implementation that simplifies
            # the MVN of the proposal distribution
            wc_delta = (theta - theta_proposal_gd) 
            wp_delta = (theta_proposal - theta_gd)

            sigma_sq = self.step_theta

            first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
            second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq

            diff_prop =  first - second
            n_langevin += 1
        else:
            diff_prop = 0
            theta_proposal = np.random.normal(theta, self.step_theta, self.theta_size)

        # eta proposal
        eta_proposal = eta + np.random.normal(0, self.step_eta, 1)
        tau_proposal = np.exp(eta_proposal)   

        # calculate the prior likelihood
        prior_proposal = self.prior_likelihood(
            sigma_squared, nu_1, nu_2, theta_proposal, tau_proposal
        )  # takes care of the gradients
        # calculate the likelihood considering observations
        [likelihood_proposal, pred_y[ii,], sim_y[ii,], rmse_data[ii]] = self.likelihood_function(
            theta_proposal, tau_proposal
        )

        # calculate the test likelihood
        [_, test_pred_y[ii,], test_sim_y[ii,], test_rmse_data[ii]] = self.likelihood_function(
            theta_proposal, tau_proposal, test=True
        )

        # since we using log scale: based on https://www.rapidtables.com/math/algebra/Logarithm.html
        diff_likelihood = likelihood_proposal - likelihood
        diff_priorlikelihood = prior_proposal - prior_likelihood
        
        mh_prob = min(1, np.exp(diff_likelihood + diff_priorlikelihood + diff_prop))

        u = np.random.uniform(0, 1)

        # Accept/reject
        if u < mh_prob:
            # Update position
            n_accept += 1
            # update
            likelihood = likelihood_proposal
            prior_likelihood = prior_proposal
            theta = theta_proposal
            eta = eta_proposal
            # and store
            pos_theta[ii,] = theta_proposal
            pos_tau[ii,] = tau_proposal
            pos_eta[ii,] = eta_proposal
        else:
            # store
            pos_theta[ii,] = pos_theta[ii-1,]
            pos_tau[ii,] = pos_tau[ii-1,]
            pos_eta[ii,] = pos_eta[ii-1,]

    # print the % of times the proposal was accepted
    accept_ratio = (n_accept / self.n_samples) * 100
    print('{:.3}% was accepted'.format(accept_ratio))

    # store the posterior of theta and tau, as well as the RMSE of these samples
    self.pos_theta = pos_theta[self.n_burnin:, ]
    self.pos_tau = pos_tau[self.n_burnin:, ]
    self.pos_eta = pos_eta[self.n_burnin:, ]
    # these are unused at present
    # self.rmse_data = rmse_data[self.n_burnin:]
    # self.test_rmse_data = test_rmse_data[self.n_burnin:]

    # Create a pandas dataframe to store the posterior samples of theta and tau, the 
    # associated RMSE
    results_dict = {'w{}'.format(_): self.pos_theta[:, _].squeeze() for _ in range(self.theta_size-2)}
    results_dict['b0'] = self.pos_theta[:, self.theta_size-2].squeeze()
    results_dict['b1'] = self.pos_theta[:, self.theta_size-1].squeeze()    
    results_dict['tau'] = self.pos_tau.squeeze()
    # results_dict['rmse'] = self.rmse_data.squeeze()
    # results_dict['test_rmse'] = self.test_rmse_data.squeeze()

    # return the predictions
    pred_dict = {}
    pred_dict['train_pred'] = pred_y[self.n_burnin:,:]
    pred_dict['train_sim'] = sim_y[self.n_burnin:,:]
    pred_dict['test_pred'] = test_pred_y[self.n_burnin:,:]
    pred_dict['test_sim'] = test_sim_y[self.n_burnin:,:]
    
    results_df = pd.DataFrame.from_dict(
        results_dict
    )

    return results_df, pred_dict

# %% [markdown]
# Below we the put together MCMC class, which runs the sampling of our model. For readability we have defined the some key funtions above and we will add these to the MCMC class below. The class contains the following functions:
# 

# %%
class MCMC:
    def __init__(self, model, n_samples, n_burnin, x_data, y_data, x_test, y_test):
        self.n_samples = n_samples # number of MCMC samples
        self.n_burnin = n_burnin # number of burn-in samples
        self.x_data = x_data # (N x num_features)
        self.y_data = y_data # (N x 1)
        self.x_test = x_test # (Nt x num_features)
        self.y_test = y_test # (Nt x 1)

        # MCMC parameters - defines how much variation you need in changes to theta, tau
        self.step_theta = 0.025;  
        self.step_eta = 0.2; # note eta is used as tau in the sampler to consider log scale.
        # Hyperpriors
        self.sigma_squared = 25
        self.nu_1 = 0
        self.nu_2 = 0

        # initisalise the linear model class
        self.model = model
        self.use_langevin_gradients = True
        self.sgd_depth = 1
        self.l_prob = 0.5 # likelihood prob
        self.theta_size = self.model.n_params # weights for each feature and a bias term

        # store output
        self.pos_theta = None
        self.pos_tau = None
        self.pos_eta = None
        self.rmse_data = None

        # functions defined above - this is poor practice, but done for readability 
        # and clarity
        if self.model.data_case == 'regression':
            self.likelihood_function = MethodType(likelihood_function, self)
            self.prior_likelihood = MethodType(prior_likelihood, self)
        elif self.model.data_case == 'classification':
            self.likelihood_function = MethodType(classification_likelihood_function, self)
            self.prior_likelihood = MethodType(classification_prior_likelihood, self)
        else:
            raise ValueError('data_case must be regression or classification')
        self.sampler = MethodType(sampler, self)

    def rmse(self, predictions, targets):
        '''
        Additional error metric - root mean square error
        '''
        return np.sqrt(((predictions - targets) ** 2).mean())


    def accuracy(self, predictions, targets):
        '''
        Additional error metric - accuracy
        '''
        count = (predictions == targets).sum()
        return 100 * (count / predictions.shape[0])

    def model_draws(self, num_draws = 10, verbose=False):
        '''
        Calculate the output of the network from draws of the posterior distribution
        Input:
            - num_draws: number of draws
            - verbose: if True, print the details of each draw
        Output:
            - pred_y: (num_draws x N) ouptut of the NN for each draw
        '''
        accuracy = np.zeros(num_draws)
        rmse_data = np.zeros(num_draws) 
        pred_y = np.zeros((num_draws, self.x_data.shape[0]))
        sim_y = np.zeros((num_draws, self.x_data.shape[0]))

        for ii in range(num_draws):
            theta_drawn = np.random.normal(self.pos_theta.mean(axis=0), self.pos_theta.std(axis=0), self.theta_size)
            # draw in the eta space to give this stability
            eta_drawn = np.random.normal(self.pos_eta.mean(), self.pos_eta.std())
            tausq_drawn = np.exp(eta_drawn)
            [likelihood_proposal, pred_y[ii,], sim_y[ii,], rmse_data[ii]] = self.likelihood_function(
                theta_drawn, tausq_drawn
            )
            if verbose:
                print(
                    'Draw {} - RMSE: {:.3f}. Theta: {}, Tau {}'.format(
                        ii, rmse_data[ii], theta_drawn, tausq_drawn
                    )
                )
        return pred_y, sim_y

# %% [markdown]
# ## Load the data
# - Load in the suspot data
# - You can also load in the other regeression datasets `Lazer` and `Energy`

# %%
# load the data
name        = "Abalone" # "Sunspot", "Abalone", "Iris", "Ionosphere"
train_data   = np.loadtxt("data/{}/train.txt".format(name))
test_data    = np.loadtxt("data/{}/test.txt".format(name))

print('Training data shape: {}'.format(train_data.shape))
print('Test data shape: {}'.format(test_data.shape))

# %% [markdown]
# ## Sample using MCMC
# 
# - Create the MCMC loop and sample the posterior distribution

# %%
## MCMC Settings and Setup
n_samples       = 5000 # number of samples to draw from the posterior
burn_in         = int(n_samples* 0.5) # number of samples to discard before recording draws from the posterior
hidden          = 10
learning_rate   = 0.01

x_data = train_data[:,:-1]
y_data = train_data[:,-1]
x_test = test_data[:,:-1]
y_test = test_data[:,-1]

if name in ['Sunspot','Abalone']:
    layer_sizes = [x_data.shape[1], hidden, 1]
    data_case = 'regression'
elif name in ['Iris']:
    layer_sizes = [x_data.shape[1], hidden, 3]
    data_case = 'classification'
elif name in ['Ionosphere']:
    layer_sizes = [x_data.shape[1], hidden, 2]
    data_case = 'classification'
else:
    raise ValueError('data_case is invalid.')
    
# Initialise the MCMC class
nn_model = NeuralNetwork(layer_sizes,learning_rate,data_case)
mcmc = MCMC(nn_model,n_samples, burn_in, x_data, y_data, x_test, y_test)

# Run the sampler
results, pred = mcmc.sampler()

# %%
# gather the predicitons into useful variables
pred_y = pred['train_pred']
sim_y = pred['train_sim']
pred_y_test = pred['test_pred']
sim_y_test = pred['test_sim']

# %% [markdown]
# ## Analyse the results
# Plot the data with the mean linear fit and some uncertainty.
# 
# Plot the posterior distribution and trace for each parameter using ipywidgets.

# %%
# plot the data with the model predictions from posterior draws
if name == "Iris":
    plot_minmax = (-0.05,2.05)
else:
    plot_minmax = (0,1)
plot_ycorr_scatter(
    y_data,
    pred_y,
    minmax=plot_minmax
)
plot_ycorr_scatter(
    y_test,
    pred_y_test,
    minmax=plot_minmax
)

# %%
plot_y_timeseries(
    y_data,
    pred_y,
    dataset_name=name + ' Train',
    ci=True
)

# %%
plot_y_timeseries(
    y_test,
    pred_y_test,
    dataset_name=name + ' Test',
    ci=True
)

# %%
# Print the train/test RMSE
if data_case == 'regression':
    print('RMSE: mean (std)')
    train_RMSE = np.array([mcmc.rmse(pred_y[_,:], y_data) for _ in np.arange(pred_y.shape[0])])
    test_RMSE = np.array([mcmc.rmse(pred_y_test[_,:], y_test) for _ in np.arange(pred_y_test.shape[0])])
    print('Train RMSE: {:.5f} ({:.5f})'.format(train_RMSE.mean(),train_RMSE.std()))
    print('Test RMSE: {:.5f} ({:.5f})'.format(test_RMSE.mean(),test_RMSE.std()))  
elif data_case == 'classification':
    print('Accuracy: mean (std)')
    train_acc = np.array([mcmc.accuracy(pred_y[_,:], y_data) for _ in np.arange(pred_y.shape[0])])
    test_acc = np.array([mcmc.accuracy(pred_y_test[_,:], y_test) for _ in np.arange(pred_y_test.shape[0])])
    print('Train Accuracy: {:.3f}% ({:.3f})'.format(train_acc.mean(),train_acc.std()))
    print('Test Accuracy: {:.3f}% ({:.3f})'.format(test_acc.mean(),test_acc.std()))

# %%
def plot_hist(results, param_name):
    # results = results_rmse
    posterior_mean = results[param_name].mean()
    print('{:.3f} mean value of posterior'.format(posterior_mean))
    histogram_trace(results[param_name].values)

# use ipywidgets to get a "gui" dropdown to view all the parameters
interact(
    plot_hist, 
    results=fixed(results), 
    param_name=widgets.Dropdown(
        options=results.columns,
        value='w0',
        description='Parameter:',
    )
)

# %%
# Visualise all the network weights
boxplot_weights(results,width=20)

# %%



