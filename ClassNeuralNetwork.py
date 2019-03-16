import numpy as np
from sklearn import preprocessing
from copy import deepcopy

class NeuralNetwork(object):
    ### init Function ###
    '''
    __init__: creates the neural network model with required parameters
            Note: 
            1 - Weights are appended with the bias term weight
            2 - Data is appended with One/bias terms at the begining, one row is one training sample, total rows
            equal to the total training set
    '''
    def __init__(self, X, d, split=0.8, act_func='Sigmoid', leak=0.2, loss='MSE', layer_info=np.array([2, 2]), 
                 lr_info=np.array([0.1, 0.5]), momentum=0.5, epsilon=4.7, maxIter=10000):
        self.X = X
        self.d = d
        self.split = split
        self.prep_data()
        # bias has been appended to the data Nx(D+1) N: no of training samples D: dimension of the data
        self.row, self.col = self.X_train.shape 
        self.num_hidden_layers = layer_info.shape[0] # number of hiddent layers 
        self.outputs = self.label_train.shape[1]
        self.W_guess = [] 
        self.M = [] # Momentum vector for each layer
        # weights from the inputs to the first hidden layer
        self.W_guess.append(np.random.normal(0, np.sqrt(2/(self.col-1+layer_info[0])), (layer_info[0], self.col))) 
        self.M.append(np.zeros((layer_info[0], self.col)))
        # generate weights between internal layers
        for i in range(self.num_hidden_layers-1): 
            self.W_guess.append(np.random.normal(0, np.sqrt(2/(layer_info[i]+layer_info[i+1])), (layer_info[i+1], layer_info[i]+1)))
            self.M.append(np.zeros((layer_info[i+1], layer_info[i]+1)))
        # weights for final layer to output layer
        self.W_guess.append(np.random.normal(0, np.sqrt(2/(layer_info[-1]+self.outputs)), (self.outputs, layer_info[-1]+1))) 
        self.M.append(np.zeros((self.outputs, layer_info[-1]+1)))
        self.W = deepcopy(self.W_guess)
        self.act_func = act_func
        self.leak = leak
        self.loss = loss
        self.eta = lr_info[0]
        self.eta_decay = lr_info[1]
        self.beta = momentum
        self.epsilon = epsilon
        self.maxIter = maxIter
        
    ### Basic Functions ###
    '''
    prep_data: prepares the data for training
    '''
    def prep_data(self):
        scaler = preprocessing.StandardScaler().fit(self.X)
        X_scaled = scaler.transform(self.X)
        bias = np.ones((X_scaled.shape[0], 1))
        X_scaled = np.hstack((bias, X_scaled))
        numsplit = int(self.X.shape[0]*self.split)
        self.X_val = X_scaled[numsplit:]
        self.X_train = X_scaled[:numsplit]
        self.label_val = self.d[numsplit:]
        self.label_train = self.d[:numsplit]      
        
    '''
    activation_func: which activation function to use for the hidden layers
    '''
    def activation_func(self, v):
        if self.act_func == 'Tanh':
            value = np.tanh(v)
        elif self.act_func == 'ReLU':
            value = np.maximum(np.full((v.shape), self.leak*v), v)
        elif self.act_func == 'Sigmoid':
            value = 1/(1+np.exp(-v))
        else:
            print('Invalid Activation function')
            value = None
        return value
    
    '''
    activation_func_derivative: derivative of the activation which is being used
    '''
    def activation_func_derivative(self, v):
        if self.act_func == 'Tanh':
            value = 1 - np.square(np.tanh(v))
        elif self.act_func == 'ReLU':
            value = deepcopy(v)
            value[value<0] = self.leak
            value[value==0] = 0.5
            value[value>0] = 1
        elif self.act_func == 'Sigmoid':
            value = (1/(1+np.exp(-v))) * (1 - (1/(1+np.exp(-v))))
        else:
            print('Invalid Activation function')
            value = None
        return value
    
    '''
    output_act_func: 
                    1 - softmax: output layer activation function is softmax if we are doing 
                        classification coupled with Cross Entropy loss
                    2 - sigmoid: output layer activation function is sigmoid if we are doing
                        classification couple with Meas Squared loss
    '''
    def output_act_func(self, v):
        if self.loss == 'CE':
            e = np.exp(v - np.max(v))
            value = e/np.sum(e)
        elif self.loss == 'MSE':
            value = 1/(1+np.exp(-v))
        else:
            print('Invalid Activation function')
            value = None
        return value

    '''
    output_layer_err_derivative: based 
    '''
    def output_layer_err_derivative(self, z, d, v):
        if self.loss == 'CE':
            value = 1/self.row * (z - d)
        elif self.loss == 'MSE':
            value = 1/self.row * (z - d) * (1/(1+np.exp(-v))) * (1 - (1/(1+np.exp(-v))))
        return value
    
    '''
    learning_rate_decay: decay the learning rate whenever the 
            1 - validation error stops to improving or 
            2 - loss increases 
    '''
    def learning_rate_decay(self, L1, L2):
        if L1 > L2:
            self.eta = self.eta * self.eta_decay
    
    ### Advanced Functions ###
    '''
    feedforward: calculates all layers induced local fields (output with no activation function) V
            and internal layer outputs (output with activation function applied plus bias term appended) Z, but for last
            layer i.e., output layer I don't add one infront. Appended bias term is not compulsory infact it shouldn't
            be there but for my formulation of backprop it is required hence I added it. It is just for functionality
            but has no physical significance because bias is not part of the output.
    '''
    def feedforward(self, x, W):
        V = []
        Z = []
        Z.append(x.reshape(self.col, 1)) # append input layer (including bias) only for intial cal finally it will be removed
        for i in range(self.num_hidden_layers): # cal V, Z for internal layers
            V.append(np.matmul(W[i], Z[-1]))
            temp = self.activation_func(V[-1])
            temp = np.concatenate(([[1]], temp), axis=0)
            Z.append(temp)
        V.append(np.matmul(W[-1], Z[-1])) # cal V, Z for output layer
        temp = self.output_act_func(V[-1])
        Z.append(temp)
        Z = Z[1:]
        return V, Z
    
    '''
    back_prop: here backward propagation is done and gradients for each layer is calculated
    '''
    def back_prop(self, x, d, W, V, Z):
        back_prop_err = []
        gradient = []
        # Backpropagated error for output layer
        delta = self.output_layer_err_derivative(Z[-1], d.reshape(self.outputs, 1), V[-1])
        back_prop_err.append(delta)
        # Backpropagated error for hidden layers
        for i in reversed(range(self.num_hidden_layers)):
            derivative = self.activation_func_derivative(V[i])
            temp = np.matmul(W[i+1][:, 1:].T, back_prop_err[-1])
            delta = np.multiply(temp, derivative)
            back_prop_err.append(delta)
        # Reverse your backpropagated error array because data was stored in reverse order
        # i.e. last layer delta is in the first element
        back_prop_err = list(reversed(back_prop_err))
        # Caculate the gradients for first hidden layers
        gradient.append(np.matmul(back_prop_err[0], x.reshape(self.col, 1).T))
        # Caculate the gradients for remaining layers
        for i in range(1, self.num_hidden_layers+1):
            gradient.append(np.matmul(back_prop_err[i], Z[i-1].T))
        return gradient
    
    '''
    update_weights: update the weights using the gradient descent with momentum method
    '''
    def update_weights(self, gradient):
        for i in range(self.num_hidden_layers+1):
            self.M[i] = (self.beta * self.M[i]) - (self.eta * gradient[i])
            self.W[i] = self.W[i] + self.M[i]
            
    '''
    cal_loss: here we calculate the cross-entropy loss and no of misclassifications
    '''
    def cal_loss(self, X, d):
        if self.loss == 'CE':
            loss_value = 0
            error = 0
            row, col = X.shape
            for i in range(row):
                v, z = self.feedforward(X[i], self.W)
                loss_value = loss_value + np.dot(-d[i], np.log(z[-1].reshape(self.outputs)).T)
                if np.argmax(d[i]) != np.argmax(z[-1]):
                    error += 1
        elif self.loss == 'MSE':
            loss_value = 0
            error = 0
            row, col = X.shape
            for i in range(row):
                v, z = self.feedforward(X[i], self.W)
                loss_value = loss_value + np.sum(np.square(d[i] - z[-1].reshape(self.outputs).T))
                if np.abs(d[i]- z[-1]) > 0.5:
                    error += 1            
        return loss_value, error
        
    '''
    train_NN: trains the neural network
    result = [epoch, loss_train, loss_val, error_train, error_val]
    '''
    def train_NN(self):
        result = []
        epoch = 0
        while epoch <= self.maxIter:
            L = np.arange(self.row)
            np.random.shuffle(L)
            for i in L:
                v, z = self.feedforward(self.X_train[i], self.W)
                gradient = self.back_prop(self.X_train[i], self.label_train[i], self.W, v, z)
                self.update_weights(gradient)
            loss_train, error_train = self.cal_loss(self.X_train, self.label_train)
            loss_val, error_val = self.cal_loss(self.X_val, self.label_val)
            result.append([epoch, loss_train, loss_val, error_train, error_val])
            print('Epoch: %d eta: %0.3f Loss_Train %0.2f Loss_Val %0.2f Error_Train %0.2f%% Error_Val %0.2f%%' 
                  % (epoch, self.eta, loss_train, loss_val, 100*(error_train/self.row), 100*(error_val/self.X_val.shape[0])))
            if epoch != 0:
                if 100*(error_val/self.X_val.shape[0]) <= self.epsilon:
                    result = np.array(result)
                    self.W_opt = deepcopy(self.W)
                    print('Optimal Weights Reached!!!!!')
                    return self.W_opt, result
                else:
                    self.learning_rate_decay(result[-1][1], result[-2][1])
                    epoch += 1
            else:
                epoch += 1
        result = np.array(result)
        self.W_opt = deepcopy(self.W)
        print('Optimal Weights Reached!!!!!')
        return self.W_opt, result
    
    '''
    output_generator: based on the loss we are using it will convert the output into readable format
    '''
    def output_generator(self, z):
        if self.loss == 'CE':
            value = np.zeros(self.outputs)
            value[np.argmax(z)] = 1
        elif self.loss == 'MSE':
            if z >= 0.5:
                value = 1
            else:
                value = 0
        return value
    
    '''
    predict: generate results on the test set
    '''
    def predict(self, X, W):
        row, col = X.shape
        scaler = preprocessing.StandardScaler().fit(self.X)
        X_scaled = scaler.transform(X)
        bias = np.ones((X_scaled.shape[0], 1))
        X_scaled = np.hstack((bias, X_scaled))
        raw_output_predict = []
        output_predict = []
        for i in range(row):
            v, z = self.feedforward(X_scaled[i], W)
            temp = self.output_generator(z[-1])
            raw_output_predict.append(z[-1])
            output_predict.append(temp)
        output_predict = np.array(output_predict)
        return raw_output_predict, output_predict
    
    '''
    predict_loss: generate results on the test set also calculates the loss based on test set labels
    '''
    def predict_loss(self, X, d, W):
        row, col = X.shape
        scaler = preprocessing.StandardScaler().fit(self.X)
        X_scaled = scaler.transform(X)
        bias = np.ones((X_scaled.shape[0], 1))
        X_scaled = np.hstack((bias, X_scaled))
        raw_output_predict = []
        output_predict = []
        for i in range(row):
            v, z = self.feedforward(X_scaled[i], W)
            temp = self.output_generator(z[-1])
            raw_output_predict.append(z[-1])
            output_predict.append(temp)
        output_predict = np.array(output_predict)
        loss, error = self.cal_loss(X_scaled, d)
        return raw_output_predict, output_predict, loss, error