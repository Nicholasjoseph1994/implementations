import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.datasets import make_moons

class NeuralNetwork():
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        np.random.seed(1)

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))
    def relu(self, x):
        def _relu(x):
            return x if x >= 0 else 0
        return np.vectorize(_relu)(x)

    def initialize_parameters(self):
        """
        This initializes the weights to small values and the biases to zero
        ARGS:
            layer_dims (List[int]) - The size of the input, hidden and output layers from input to output

        RETURNS:
            None - Initializes instance variables
        """

        self.params = {}
        L = len(self.layer_dims)

        #TODO: set this to sqrt( 2 / m_(i-1)) with m_(i-1) as the number of features in the last layer
        weight_size = 0.1

        # Initialize weights. W1 is the weight matrix from layer 0 to layer 1
        for l in range(1, L):
            in_dim = self.layer_dims[l-1] 
            out_dim = self.layer_dims[l]
            self.params['W%d' % l] = np.random.randn(in_dim, out_dim) * weight_size

            assert self.params['W%d' % l].shape == (in_dim, out_dim)
    def forward_propagate(self, X):
        # Initialize layers as empty
        self.layer_activations = []
        self.layer_scores = []

        # X represents the initial activations
        A = X
        L = len(self.layer_dims)

        for l in range(1, L):
            # Save earlier activations
            A_prev = A

            # Use relu for hidden layers and sigmoid for output
            activation_func = self.sigmoid if l == L - 1 else self.relu

            # Apply linear transformation followed by the activation function
            Z = A_prev @ self.params['W%d' % l]
            A = activation_func(Z)
            assert A.shape == (X.shape[0], self.layer_dims[l])

            # Save the scores and activations for the backwards pass
            self.layer_scores.append(Z)
            self.layer_activations.append(A_prev)
        assert len(self.layer_activations) == len(self.layer_scores)

        # Should have all activations except the final one
        assert len(self.layer_activations) == len(self.layer_dims) - 1

        # Should have 1 prediction for each observation
        assert A.shape == (X.shape[0], 1)
        return A

    def backward_propagate(self, output, y):
        n = y.shape[0]
        def backward_activation(dA, l, activation):
            """
            ARGS:
                dA - The derivative of the cost function with respect to the activation layer
                l - the layer index for dA

            RETURNS:
                dZ - the derivative of the cost function with respect to the output of the linear transformation
            """
            # For a layer of size L, there should be L activations for each observation
            assert dA.shape == (n, self.layer_dims[l])

            if activation == 'sigmoid':
                # dA times the derivative of the sigmoid
                activations = self.sigmoid(self.layer_scores[l-1])
                dZ = dA * activations * (1 - activations)
            elif activation == 'relu':
                # dA times derivative of the relu
                dZ = dA * (self.layer_scores[l-1] >= 0)
            else:
                print('Only sigmoid and relu activation functions implemented so far')
                raise Exception

            assert dZ.shape == (n, self.layer_dims[l])
            return dZ
        def backward_linear(dZ, l):
            """
            ARGS:
                dZ - the derivative of the cost function with respect to the output of the linear transformation
                l - the layer index for dZ

            RETURNS:
                dA_prev - the derivative of the cost function with respect to the previous activation layer
                dW - the derivative of the cost function with respect to the weights
            """
            assert dZ.shape == (n, self.layer_dims[l])

            # Saved variables from forward propagation
            A_prev = self.layer_activations[l-1] # Shape: (n, layer_dims[l-1])
            assert A_prev.shape == (n, self.layer_dims[l-1])

            W = self.params['W%d'%l] # Shape: (layer_dims[l-1], layer_dims[l])
            assert W.shape == (self.layer_dims[l-1], self.layer_dims[l])

            # Compute derivatives
            dW = A_prev.T @ dZ / A_prev.shape[1]
            assert dW.shape == W.shape

            dA_prev = dZ @ W.T
            assert dA_prev.shape == A_prev.shape

            return (dA_prev, dW)

        def backward_layer(dA, l):
            """
            ARGS:
                dA - The derivative of the cost function with respect to the activation layer
                l - the layer index for dA

            RETURNS:
                dA_prev - the derivative of the cost function with respect to the previous activation layer
                dW - the derivative of the cost function with respect to the weights
             """

             # Use a sigmoid activation for the output layer and relu for the rest
            activation = 'sigmoid' if l == self.num_layers - 1 else 'relu'

            dZ = backward_activation(dA, l, activation)
            # print(dZ)
            # assert activation == 'sigmoid'
            return backward_linear(dZ, l)

        dAL = - (y / output) + (1 - y) / (1 - output)


        L = self.num_layers - 1 # Could be len(caches)
        self.grads = {}

        self.grads['dA%d' % (L-1)], self.grads['dW%d' % L] = backward_layer(dAL, L)

        for l in reversed(range(1, L)):
            dA_prev, dW = backward_layer(self.grads['dA%d' % l], l)
            # print(dW)
            self.grads['dA%d' % (l-1)] = dA_prev
            self.grads['dW%d' % l] = dW


    def gradient_descent(self, learning_rate):
        """
        This currently implements gradient descent
        """
        L = self.num_layers
        for l in range(1, L):
            self.params['W%d' % l] -= learning_rate * self.grads['dW%d' % l]

    def compute_cost(self, pred, y):
        m = y.shape[1]
        cost = -float(np.sum(y * np.log(pred) + (1-y) * np.log(1-pred))) / m

        # assert cost.shape == ()
        return cost

    def fit(self, X, y, learning_rate = 0.1, batch_size = 32, num_epochs = 1000, verbose = True):
        """
        This fits the neural network model to the training data
        ARGS:
            X (2d numpy array) - A matrix of features where each row is an observation and each column is a feature
            y (1d numpy array) - The labels corresponding to each row of X

        RETURNS:
            None - sets up instance variables for predict
        """
        def get_minibatches(X, y, batch_size):
            """
            ARGS:
                X (n_obs x n_features numpy array) - A matrix of features where each row is an observation and each column is a feature
                y (1d numpy array) - The labels corresponding to each obs of X
                batch_size (int) - The size of each batch

            RETURNS:
                List[tuple] - A list of (X, y) pairs where each X and y have batch_size observations
             """
            assert X.shape[0] == y.shape[0]
            batches= [(X[i: i + batch_size], y[i: i + batch_size])
                         for i in range(0, len(y), batch_size)]
            return batches

        self.initialize_parameters()
        for epoch in range(num_epochs):
            for batch_X, batch_y in get_minibatches(X, y, batch_size):

                # Propagates the input through the neural network to compute a set of predictions and caches the scores and activations
                pred = self.forward_propagate(batch_X)
                
                # Computes the gradients 
                self.backward_propagate(pred, batch_y)
                
                # Uses the gradient to update the parameter weights
                self.gradient_descent(learning_rate)

            if epoch % 100 == 0:
                full_pred = self.forward_propagate(X)
                cost = self.compute_cost(full_pred, y)
                print('Cost at epoch %d is %d' % (epoch, cost))




    def predict(self, X):
        """
        This uses a fitted NeuralNetwork model to predict the classes in X
        ARGS:
            X (2d numpy array) - A matrix of features where each row is an observation and each column is a feature

        RETURNS:
            y (1d numpy array) - The labels corresponding to each row of X

        """
        return self.forward_propagate(X)
def test():
    # Create random seed
    np.random.seed(1)

    # Create moons testing data
    X, y = make_moons(n_samples = 5000, noise = 0.1)
    y = y[:, None]
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Split into training and test
    split = len(X) // 2
    train_X = X[:split]
    test_X = X[split:]
    train_y = y[:split]
    test_y = y[split:]

    # Fit my model on training set
    clf = NeuralNetwork([X.shape[1], 3, 1])
    clf.fit(train_X, train_y)

    # Make predictions on test set
    preds = clf.predict(test_X) > 0.5

    # Evaluate accuracy
    from sklearn.metrics import accuracy_score
    print('Accuracy is {}'.format(accuracy_score(test_y, preds)))

    # Compare to sklearn Logistic Regression implementation
    from sklearn import linear_model
    skclf = linear_model.LogisticRegression(C = 1E15)
    skclf.fit(train_X, np.ravel(train_y))
    skpreds = skclf.predict(test_X) > 0.5
    print('SkLearn accuracy is {}'.format(accuracy_score(test_y, skpreds)))


test()