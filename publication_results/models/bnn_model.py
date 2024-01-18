import numpy as np
import pandas as pd

################################################################################
################################################################################


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

        self.data_case = data_case

    ################################################################################
    ################################################################################

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

    ################################################################################
    ################################################################################

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

    ################################################################################
    ################################################################################

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

    ################################################################################
    ################################################################################

    # Helper functions
    def sigmoid(self, x):
        '''
        Implentation of the sigmoid function
        '''
        return 1 / (1 + np.exp(-x))
    
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
       
    ################################################################################
    ################################################################################

       
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

    ################################################################################
    ################################################################################

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

    ################################################################################
    ################################################################################

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


################################################################################
################################################################################
