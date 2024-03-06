import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm

################################################################################
################################################################################
# Common functions

class MCMC(ABC):
    @abstractmethod
    def __init__():
        raise NotImplemented(f"{type(self).__name__}: init not implemented")
    
    @abstractmethod
    def sampler():
        raise NotImplemented(f"{type(self).__name__}: sampler not implemented")

    ################################################################################
    ################################################################################

    def model_draws(self, num_samples = 10):
        '''
        Simulate new model predictions (mu) under the assumption that our posteriors are 
        Gaussian.
        '''
        # num_samples x num_data_points
        pred_y = np.zeros((num_samples,self.x_data.shape[0]))
        sim_y = np.zeros((num_samples,self.x_data.shape[0]))

        for ii in range(num_samples):
            theta_drawn = np.random.normal(self.pos_theta.mean(axis=0), self.pos_theta.std(axis=0), self.theta_size)
            # draw in the eta space to give this stability
            eta_drawn = np.random.normal(self.pos_eta.mean(), self.pos_eta.std())
            tausq_drawn = np.exp(eta_drawn)
            [_, pred_y[ii,:], sim_y[ii,:],_] = self.likelihood_function(
                theta_drawn, tausq_drawn
            )
        return pred_y, sim_y

    ################################################################################
    ################################################################################

    @staticmethod
    def rmse(predictions, targets):
        '''
        Additional error metric - root mean square error
        '''
        return np.sqrt(((predictions - targets) ** 2).mean())

    ################################################################################
    ################################################################################

    @staticmethod
    def r2(predictions, targets):
        '''
        Additional error metric - R^2
        '''
        ssr = np.sum((predictions - targets)**2)
        sst = np.sum((targets - np.mean(targets))**2)
        return 1 - (ssr / sst)

    ################################################################################
    ################################################################################

    @staticmethod
    def accuracy(predictions, targets):
        '''
        Additional error metric - accuracy
        '''
        count = (predictions == targets).sum()
        return 100 * (count / predictions.shape[0])
    
    ################################################################################
    ################################################################################

    # Define the likelihood function
    def regression_likelihood_function(self, theta, tausq, test=False):
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
        # first make a prediction with parameters theta
        model_prediction, _ = self.model.evaluate_proposal(x_data, theta)
        model_simulation = model_prediction + np.random.normal(0,tausq,size=model_prediction.shape) 
        accuracy = self.rmse(model_prediction, y_data) #RMSE error metric 
        # now calculate the log likelihood
        log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(y_data - model_prediction) / tausq)
        return [log_likelihood, model_prediction, model_simulation, accuracy] 

    ################################################################################
    ################################################################################

    # Define the prior
    def regression_prior_likelihood(self, sigma_squared, nu_1, nu_2, theta, tausq): 
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
    
    ################################################################################
    ################################################################################

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

    ################################################################################
    ################################################################################

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
        
    ################################################################################
    ################################################################################

        
    ################################################################################
    ################################################################################

     

################################################################################
################################################################################
# MCMC for Linear Regression
################################################################################
################################################################################

class MCMC_Linear(MCMC):
    def __init__(self, model, n_samples, n_burnin, x_data, y_data, x_test, y_test):
        self.n_samples = n_samples # number of MCMC samples
        self.n_burnin = n_burnin # number of burn-in samples
        self.x_data = x_data # (N x M)
        self.y_data = y_data # (N x 1)
        self.x_test = x_test # (Nt x num_features)
        self.y_test = y_test # (Nt x 1)

        # MCMC parameters - defines the variance term in our Gaussian random walk
        self.step_theta = 0.02;  
        self.step_eta = 0.01; # note eta is used as tau in the sampler to consider log scale.  
        
        # model hyperparameters
        # considered by looking at distribution of  similar trained  models - i.e distribution of weights and bias
        self.sigma_squared = 5
        self.nu_1 = 0
        self.nu_2 = 0

        # initisalise the linear model class
        self.model = model
        self.theta_size = self.model.n_params # per the linear model

        # store output
        self.pos_theta = None
        self.pos_tau = None
        self.pos_eta = None
        self.rmse_data = None

        # functions defined above - this is poor practice, but done for readability 
        # and clarity
        if self.model.data_case == 'regression':
            self.likelihood_function = self.regression_likelihood_function
            self.prior_likelihood = self.regression_prior_likelihood
        elif self.model.data_case == 'classification':
            self.likelihood_function = self.classification_likelihood_function
            self.prior_likelihood = self.classification_prior_likelihood
        else:
            raise ValueError('data_case must be regression or classification')
        
    ################################################################################
    ################################################################################

    def sampler(self):
        '''
        Run the sampler for a defined linear model
        '''
        ## Define empty arrays to store the sampled posterior values
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
        # initialise theta - the model parameters
        theta = np.random.randn(self.theta_size)
        # make initial prediction
        pred_y[0,], _ = self.model.evaluate_proposal(self.x_data, theta)

        # initialise eta - we sample eta as a gaussian random walk in the log space of tau^2
        eta = np.log(np.var(pred_y[0,] - self.y_data))
        tausq_proposal = np.exp(eta)

        # calculate the prior likelihood
        prior_likelihood = self.prior_likelihood(self.sigma_squared, self.nu_1, self.nu_2, theta, tausq_proposal)
        # calculate the likelihood considering observations
        [likelihood, pred_y[0,], sim_y[0,], rmse_data[0]] = self.likelihood_function(theta, tausq_proposal)

        n_accept = 0  
        ## Run the MCMC sample for n_samples
        for ii in tqdm(np.arange(1,self.n_samples),miniters=np.int64(self.n_samples/20)):
            # Sample new values for theta and tau using a Gaussian random walk
            theta_proposal = theta + np.random.normal(0, self.step_theta, self.theta_size)
            eta_proposal = eta + np.random.normal(0, self.step_eta, 1) # sample tau^2 in log space
            tausq_proposal = np.exp(eta_proposal)   

            # calculate the prior likelihood
            prior_proposal = self.prior_likelihood(
                self.sigma_squared, self.nu_1, self.nu_2, theta_proposal, tausq_proposal
            )
            # calculate the likelihood considering observations
            [likelihood_proposal, pred_y[ii,], sim_y[ii,], rmse_data[ii]] = self.likelihood_function(
                theta_proposal, tausq_proposal
            )

            # calculate the test likelihood
            [_, test_pred_y[ii,], test_sim_y[ii,], test_rmse_data[ii]] = self.likelihood_function(
                theta_proposal, tausq_proposal, test=True
            )

            # Noting that likelihood_function and prior_likelihood return log likelihoods,
            # we can use log laws to calculate the acceptance probability
            diff_likelihood = likelihood_proposal - likelihood
            diff_priorlikelihood = prior_proposal - prior_likelihood

            mh_prob = min(1, np.exp(diff_likelihood + diff_priorlikelihood))

            # sample to accept or reject the proposal according to the acceptance probability
            u = np.random.uniform(0, 1)
            if u < mh_prob:
                # accept and update the values
                n_accept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_proposal
                theta = theta_proposal
                eta = eta_proposal
                # store to make up the posterior
                pos_theta[ii,] = theta_proposal
                pos_tau[ii,] = tausq_proposal
                pos_eta[ii,] = eta_proposal
            else:
                # reject move and store the old values
                pos_theta[ii,] = pos_theta[ii-1,]
                pos_tau[ii,] = pos_tau[ii-1,]
                pos_eta[ii,] = pos_eta[ii-1,]

        # calculate the acceptance rate as a check
        accept_rate = (n_accept / self.n_samples) * 100
        print('{:.3f}% were accepted'.format(accept_rate))

        # store the posterior (samples after burn in) in a pandas dataframe and return
        self.pos_theta = pos_theta[self.n_burnin:, ]
        self.pos_tau = pos_tau[self.n_burnin:, ] 
        self.pos_eta = pos_eta[self.n_burnin:, ]
        # self.rmse_data = rmse_data[self.n_burnin:]

        # split theta into w and b
        results_dict = {'w{}'.format(_): self.pos_theta[:, _].squeeze() for _ in range(self.theta_size-1)}
        results_dict['b'] = self.pos_theta[:, -1].squeeze()
        results_dict['tau'] = self.pos_tau.squeeze()
        # results_dict['rmse'] = self.rmse_data.squeeze()

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

################################################################################
################################################################################
# MCMC for BNN
################################################################################
################################################################################

class MCMC_BNN(MCMC):
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
            self.likelihood_function = self.regression_likelihood_function
            self.prior_likelihood = self.regression_prior_likelihood
        elif self.model.data_case == 'classification':
            self.likelihood_function = self.classification_likelihood_function
            self.prior_likelihood = self.classification_prior_likelihood
        else:
            raise ValueError('data_case must be regression or classification')

    ################################################################################
    ################################################################################


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
        for ii in tqdm(np.arange(1,self.n_samples),miniters=np.int64(self.n_samples/20)):
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
        print('{:.3}% were acepted'.format(accept_ratio))

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

################################################################################
################################################################################
