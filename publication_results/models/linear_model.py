import numpy as np
import pandas as pd

################################################################################
################################################################################

class LinearModel:
    '''
    Simple linear model with a single output (y) given the covariates x_1...x_M of the form:
    y = w_1 * x_1 + ... + w_M * x_M + b
    where M = number of features, w are the weights, and b is the bias.
    '''
    # Initialise values of model parameters
    def __init__(self, layer_sizes, data_case='regression'):
        self.w = None
        self.b = None 
        self.input_num = layer_sizes[0]
        self.output_num = layer_sizes[-1]
        self.n_params = (self.input_num * self.output_num) + self.output_num
        self.data_case = data_case

    ################################################################################
    ################################################################################

    # Function to take in data and parameter sample and return the prediction
    def evaluate_proposal(self, data, theta):
        '''
        Encode the proposed parameters and then use the model to predict
        Input:
            data: (N x M) array of data
            theta: (M + 1) vector of parameters. The last element of theta consitutes the bias term (giving M + 1 elements)
        '''
        self.encode(theta)  # method to encode w and b
        prediction = self.predict(data) # predict and return
        if self.data_case == 'classification':
            # probs = np.vstack([1 - prediction, prediction]).T
            probs = self.softmax(prediction)
            prediction = np.argmax(probs, axis=1)
        else:
            probs = None
            prediction = prediction.squeeze()
        return prediction, probs
    
    ################################################################################
    ################################################################################

    # Linear model prediction
    def predict(self, x_in):
        y_out = x_in.dot(self.w) + self.b 
        # if self.data_case == 'classification':
        #     y_out = self.sigmoid(y_out)
        return y_out

    ################################################################################
    ################################################################################
 
    def softmax(self, x):
        '''
        Implentation of the softmax function
        '''
        prob = np.exp(x) / np.sum(np.exp(x))
        return prob

    ################################################################################
    ################################################################################

    # Helper function to split the parameter vector into w and band store in the model
    def encode(self, theta):
        self.w =  np.reshape(theta[:-self.output_num], (-1, self.output_num))
        self.b = theta[-self.output_num:] 


################################################################################
################################################################################
