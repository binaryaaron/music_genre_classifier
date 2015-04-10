"""
classify.py handles implementation of the meaty parts of the classifier -
estimation of priors and predicting new data.
"""
try:
    import numpy as np
    from scipy.special import expit
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
    Makes a np array that satisfies the \delta function, where
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


def _logistic(w, x=None):

    def matrix_log():
        # negative exponent?
        # print('matrix logistic')
        return logistic(np.dot(w, x))
    def single_log():
        # print('single logistic')
        return sigmoid(w)

    if x is not None:
        return matrix_log()
    else:
        return single_log()

def sigmoid(x):
    """
    Wrapper for scipy's sigmoid function that returns a nice, quick
    1/(1 + exp(-x))
    Args:
        v (anything!): a vector or a number, hopefully.
    Returns:
        sigmoid-ized version of your input.
    """
    return expit(x)


def logistic(x):
    return np.exp(-x) / (1 + np.exp(-x))


def sig_prime(x):
    """
    First derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def _logsumexp(X, Y):
    """
    handy calculator for the log sum exp trick
    """
    return np.log(np.exp(np.dot(X, Y)))


def P(X, W):
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
    # denominator = np.sum(np.log(sigmoid(np.dot(W, X.T))), axis=0)
    # denominator = 1 + np.sum(sigmoid(np.dot(W, X.T)), axis=0)
    P = np.dot(W, X.T)
    P[K-1] = 1
    colsums = np.sum(P, axis=0)
    P = P/colsums
    # print(P)
    return np.exp(P)


def grad_decent(X, W, Delta, _lambda=None, eta=0.0001, rounds=20, debug=False):
    """
    A hacky implementation of gradient descent. 
    Args:
        X (numpy array): data matrix with a single column of ones
        W (numpy array): weight matrix that is to be updated
        _lambda (float): penalty rate for logistic regression
        eta (float): learning rate for gradient descent
        rounds (int): number of training rounds
        debug (boolean): flag to turn on printing

    Returns:
        W
    """
    if _lambda is None:
        _lambda = 5
    error = 0

    for i in range(rounds):
        _delta = Delta - np.log(P(X, W))
        d = np.dot(_delta, X)
        # gets the largest deviation
        _error = np.abs(d).max()
        if _error > error:
            eta /= (1 + i%3/rounds)
            print('Step %d: Error: %f \n \t updating learning rate: %f' % (i, error, eta))
        error = _error
        f = d - (_lambda * W)
        W += (eta * f)

    print('Error: %f \n Learn rate: %f' % (error, eta))
    return W


def train():
    
    W = 0 + np.zeros(shape=(len(classes), ffts.shape[1]), dtype='float64')
    W.T[0] = 1
    return classify.grad_decent(ffts, W, Delta, 
                                _lambda=0.001, 
                                eta = 0.001, 
                                rounds = 1000, 
                                debug=True)


def prediction(W, X_new):
    """
    Returns class predictions for a matrix of test data.
    Args:
    W (numpy array): trained matrix of weights
    X_new (predicted data): matrix of unseen weights
    """
    K = X_new.shape[0]
    pred_classes = np.zeros((K, W.shape[0])) - 1

    for k in range(K-1):
        pred_classes[k,:] = _logistic(np.dot(W, X_new[k].T))
    return pred_classes.argmax(axis=1), pred_classes

