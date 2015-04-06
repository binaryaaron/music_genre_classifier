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
        if debug:
            print(lab + ': ' + str(class_priors[i]))
    return class_priors


def logit(vector):
    """
    Returns the logistic transformed version of an input vector
    """
    return 1 / (1 + np.exp(-vector))


def _cond_log_like(x, labels, weights, lambda_):
    """
    Helper function that returns a single vector of log like?
    """



def cond_log_like(data, labels, weights, lambda_=0.01);
    """Gives us the conditional log likelihood for the train data, including a
    penalty term for regularization.
    Args:
        data (np.array): 2-d numpy array of training data
        labels (np.array): 1-d numpy array of class labels for training data
        weights (np.array): 2-d numpy array of training weights for the data
        lambda_ (float): number in [0, 1] as a regularization param
    Returns:
        np.array of 
    """
    def prealloc(data, labels):
        ''' Preallocate np array '''

def train(data, labels, lambda_=0.01, eta=0.001):
    """
    Main function that trains the classifer given some data, labels, and
    params.
    """

