import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
class LogisticRegression():
    def __init__(self, verbose = False):
        self.verbose = verbose

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))
    def fit(self, X, y, learning_rate = 0.01, batch_size = 32, num_epochs = 1000):
        """
        This fits the neural network model to the training data
        ARGS:
            X (n_obs x n_features numpy array) - A matrix of features where each row is an observation and each column is a feature
            y (1d numpy array) - The labels corresponding to each obs of X

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

        # Add a column of 1s to X for the bias
        n = X.shape[0]
        y = y[:, None]

        # Initialize weights and bias to a small random number
        self.weights = (np.random.randn(X.shape[1]) * 0.01)[:, None]
        assert self.weights.shape == (X.shape[1], 1)

        for epoch in xrange(num_epochs):
            # Run stochastic gradient descent
            for batch_X, batch_y in get_minibatches(X, y, batch_size):
                # Compute predictions
                pred = self.sigmoid(np.matmul(batch_X,  self.weights))
                assert pred.shape == batch_y.shape

                # Compute gradient
                dw = np.matmul(batch_X.T, (pred - batch_y))
                assert dw.shape == self.weights.shape

                # Update weights
                self.weights -= dw * learning_rate * batch_size / n

            # Print out the loss every 100 epoch's
            if epoch % 100 == 0 and self.verbose:
                pred = self.predict(X)
                loss = -np.sum(y * np.log(pred) + (1-y) * np.log(1-pred)) / n
                print('Epoch {}: loss is {}'.format(epoch, loss))
    def predict(self, X):
        """
        This uses a fitted LogisticRegression model to predict the classes in X
        ARGS:
            X (2d numpy array) - A matrix of features where each row is an observation and each column is a feature

        RETURNS:
            y (1d numpy array) - The labels corresponding to each row of X

        """
        return self.sigmoid(np.matmul(X, self.weights))
def test():
    # Create random seed
    np.random.seed(1)

    # Create moons testing data
    X, y = make_moons(n_samples = 3000, noise = 0.5)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Split into training and test
    split = len(X) / 2
    train_X = X[:split]
    test_X = X[split:]
    train_y = y[:split]
    test_y = y[split:]

    # Fit my model on training set
    clf = LogisticRegression()
    clf.fit(train_X, train_y)

    # Make predictions on test set
    preds = clf.predict(test_X) > 0.5

    # Evaluate accuracy
    from sklearn.metrics import accuracy_score
    print('Accuracy is {}'.format(accuracy_score(test_y, preds)))

    # Compare to sklearn Logistic Regression implementation
    from sklearn import linear_model
    skclf = linear_model.LogisticRegression(C = 1E15)
    skclf.fit(train_X, train_y)
    skpreds = skclf.predict(test_X) > 0.5
    print('SkLearn accuracy is {}'.format(accuracy_score(test_y, skpreds)))

test()