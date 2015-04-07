"""
classify.py handles implementation of the meaty parts of the classifier -
estimation of priors and predicting new data.
"""
try:
    import numpy as np
    import utils
except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


def est_class_priors(class_labels, classes, debug=False):
    """calculates the class priors
    Args:
        class_labels (np.array): the long list of classes for each document in
    the dataset
        classes(dict): dict of 'label', 'int' values for class labels
    Returns:
        np.array of prior counts for all
    """
    if debug:
        print('Estimating prior class probabilities')
    class_priors = np.zeros(shape=(len(classes)))
    for k, v in classes.items():
        view = class_labels[class_labels == v]
        class_priors[v] = view.size/class_labels.size
    return class_priors


def make_Delta(class_labels, n_classes):
    """
    Makes a np array that satifies the \delta function, where
    Delta(i,j) = 1 if j = i and 0 otherwise
    Args:
        class_labels (np.array): np array of the class labels
        n_classes (int): integer number of class labels
    Returns:
        n_classes x n_samples np.array
    """
    # initialize to zeros
    Delta = np.zeros(shape=(n_classes, class_labels.shape[0]))
    # may not be ness; i think broadcasting reallocates but just for now this
    # is fine
    _label = np.copy(class_labels)
    # broadcast labels to each row
    Delta[:, ] = _label
    for i in range(Delta.shape[0]):
        # where says get all indices of the array where boolean condition is
        # true and then if true, 1st arg, false, 2nd arg. If the elements in
        # the current row (corresponding to class label number) are correctly
        # placed, make them 1; 0 otherwise
        Delta[i, ] = np.where(Delta[i, :] == i, 1, 0)
    return Delta


def _logistic(w, x):
    # negative exponent?
    return np.exp(np.dot(w, x))


def logistic(X, W):
    """
    Calculates the logistic transform (i.e., probabilities) used in training
    the model.
    Args:
        X (np.array) 2-d numpy array of features
        W (np.array) 2d numpy array of weights
    """
    # number of classes
    K = W.shape[0]
    P = np.zeros(shape=(K, X.shape[0]))
    denominator = np.sum(_logistic(W, X.T), axis=0)
    for k in range(K-1):
        P[k] = _logistic(W[k], X[k].T) / 1 + denominator
    P[K-1] = 1
    colsums = np.sum(P, axis=0)
    P = P/colsums
    return P


def grad_decent(X, W, Delta, _lambda=None, eta=0.0001, rounds=100, debug=False):
    if _lambda is None:
        _lambda = 5
    error = 20
    _delta = np.zeros(shape=Delta.shape)
    d = np.zeros(shape=_delta.shape)
    f = np.zeros(shape=d.shape)

    for i in range(rounds):
        _delta = Delta - logistic(X, W)
        d = np.dot(_delta, X)
        # gets the largest deviation
        _error = np.abs(d).max()
        if _error > error:
            eta *= 0.8
        error = _error
        if debug: 
            print('Error: %f \n Learn rate: %f' % (error, eta))
        f = d - (eta * _lambda * W)
        W = W + eta * f

    print('Error: %f \n Learn rate: %f' % (error, eta))
    return W




def train(data, labels, lambda_=0.01, eta=0.001):
    """
    Main function that trains the classifer given some data, labels, and
    params.
    """

