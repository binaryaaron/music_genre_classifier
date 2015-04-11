"""
LogisticRegressionClassifer.py handles implementation of the meaty parts of the
classifier - estimation of priors and predicting new data.
"""

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


try:
    import sys
    import numpy as np
    from scipy.special import expit
    from sklearn.cross_validation import KFold
    import sklearn.metrics as metrics
    if sys.version_info < (3, 2):
        # python too old, kill the script
        sys.exit("This script requires Python 3.2 or newer! exiting.")
except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")


class LogisticRegressionClassifier(object):
    """
    Main class for the classifer.
    """

    def __init__(self, X, y, class_dict, eta=0.001, _lambda=0.001):
        """
        Creates a new executor object and initializes the main
        components.
        Args:
            Data:
                X: 2d numpy array of data. row is a sample, column is a feature
                y: 1d numpy array of data labels
                X_test: testing data, same number of columns as X
                y_test: 1d numpy array of data labels for the testing set
                class_labels: dictionary of {'class': x} for lookups

            Parameters needed by log.regression:
                Delta: indicator function; 2d numpy array
                W: Weights to be learned by the classifer
                eta: learning rate for gradient descent
                _lambda: logistic reg regularization / penalty term
                P: matrix for P(Y|X,w)
            Other:
                fitted_model: the fitted model training weights?

        """
        # ########## Data #######
        self.X = X
        self.y = y
        self._X = None
        self._y = None
        self.X_test = None
        self.y_test = None
        self.class_dict = class_dict

        # ######### Parameters needed by log.regression
        self.W = None
        self.Delta = None
        self.eta = eta
        self._lambda = _lambda
        self.error = -1
        # P = None
        # self.init_weights()
        # self.make_Delta()
        self.metrics = {'train_rounds': 0}

    def make_Delta(self):
        """
        Makes a np array that satisfies the \delta function, where
        Delta(i,j) = 1 if j = i and 0 otherwise
        """
        # initialize to zeros
        self.Delta = np.zeros(shape=(len(self.class_dict), self.y.shape[0]))
        # may not be ness; i think broadcasting reallocates but just
        # is fine for now this
        _label = np.copy(self.y)
        # broadcast labels to each row
        self.Delta[:, ] = _label
        for i in range(self.Delta.shape[0]):
            # where says get all indices of the array where boolean condition
            # is true and then if true, 1st arg, false, 2nd arg. If the
            # elements in the current row (corresponding to class label number)
            # are correctly placed, make them 1; 0 otherwise
            self.Delta[i, ] = np.where(self.Delta[i, :] == i, 1, 0)

    def _logistic(self, w, x=None):
        def matrix_log():
            # negative exponent?
            # print('matrix logistic')
            return self.logistic(np.dot(w, x))

        def single_log():
            # print('single logistic')
            return self.sigmoid(w)

        if x is not None:
            return matrix_log()
        else:
            return single_log()

    def sigmoid(self, x):
        """
        Wrapper for scipy's sigmoid function that returns a nice, quick
        1/(1 + exp(-x))
        Args:
            v (anything!): a vector or a number, hopefully.
        Returns:
            sigmoid-ized version of your input.
        """
        return expit(x)

    def logistic(self, x):
        return np.exp(-x) / (1 + np.exp(-x))

    def p_hat(self):
        """
        Calculates the logistic transform (i.e., probabilities) used in
        training the model.
        Args:
            X (np.array) 2-d numpy array of features
            W (np.array) 2d numpy array of weights
        """
        # number of classes
        K = self.W.shape[0]
        P = np.exp(np.dot(self.W, self.X.T))
        # P[K-1] = 1
        # P = np.exp(P)
        return P / np.sum(P, axis=0)

    def grad_decent(self, rounds, printing=True):
        """
        A hacky implementation of gradient descent. Implements it on class
        variables, updating the weight matrix.
        Args:
            rounds (int): number of training rounds
        Returns:
        """
        _lambda = self._lambda

        for i in range(rounds):
            d = np.dot((self.Delta - self.p_hat()), self.X)
            # gets the largest deviation
            _error = np.mean(abs(d)).max()
            if _error > self.error:
                # update learning rate more harshly
                self.eta /= (1 + (i % 20 / rounds))
                if printing:
                    print('''Step %d: Error: %f updating learning rate: %f'''
                          % (i, self.error, self.eta))
            self.error = _error
            self.W += (self.eta * (d - (_lambda * self.W)))
            self.metrics['train_rounds'] += 1

        print('Final Step %d: Error: %f \n Learn rate: %f' % (rounds,
                                                              self.error,
                                                              self.eta))

    def _cv(self, train, test):
        """
        Sets the internal indices.
        """
        self._X = self.X
        self._y = self.y
        self.X = self._X[train]
        self.X_test = self._X[test]
        self.y = self._y[train]
        self.y_test = self._y[test]
        self.error = -1
        self._lambda = 0.001
        self.eta = 0.001
        self.init_weights()
        self.metrics['train_rounds'] = 0
        self.make_Delta()

    def cross_validate(self, k=10, _print=True):
        """
        Performs several types of cross validation using the tools from
        sklearn.
        args:
            cv (sklearn scrossvalidate object): any valid CV object.
        """
        names = {v:k for k,v in self.class_dict.items()}
        names = list(zip(*names.items()))[1]

        self.metrics['accuracy'] = []
        self.metrics['cm'] = []

        kf = KFold(self.X.shape[0], n_folds=k, shuffle=True)

        for i, tup in enumerate(kf):
            train, test = tup
            self._cv(train, test)
            print("Training cross validation round %d" % i)
            print("----------------------------------")
            if i == 0:
                self.grad_decent(rounds=1000, printing=True)
            else:
                self.grad_decent(rounds=1000, printing=False)
            pred, probs = self.prediction(self.W, self.X_test)
            print("classification report ")
            print("----------------------------------")
            print(metrics.classification_report(self.y_test,
                                                pred,
                                                target_names=names))
            print("Confusion matrix")
            print("----------------------------------")
            cm = metrics.confusion_matrix(self.y_test, pred)
            print(cm)
            print("----------------------------------")
            self.X = self._X
            self.y = self._y
            self.metrics['accuracy'].append(metrics.accuracy_score(
                                            self.y_test, pred))
            self.metrics['cm'].append(cm)
        print("-------------")
        print("""we love confusion_matrices. here is the average for
              the whole training run.""")

        report = np.zeros((10, 10))
        for cm in self.metrics['cm']:
            report += cm
        report /= len(self.metrics['cm'])
        self.metrics['cv_average'] = np.floor(report)
        print(self.metrics['cv_average'])

        print("""we love metrics.Here is the average accuracy for the CV
        runs.""")
        print(np.asarray(self.metrics['accuracy']).mean())

    def train(self, rounds=1000, cv=False,
              reset=False, eta=None, lambda_=None):
        """
        Trains the model on the currently held data. This was entirely for
        personal testing.
        """

        if eta is not None:
            self.eta = eta

        if lambda_ is not None:
            self._lambda = lambda_

        if self.W is None or reset is True:
            print('initializing weight and delta matrix for the first time')
            self.error = -1
            self.init_weights()
            self.metrics['train_rounds'] = 0
            # self.Delta = make_Delta(self.class_dict, len(self.y))

        if cv is False:
            self.grad_decent(rounds=rounds)
        else:
            self.cross_validate()
        return

    def init_weights(self):
        self.W = 0.01 + np.zeros(shape=(len(self.class_dict),
                                 self.X.shape[1]),
                                 dtype='float64')
        # put all ones in the first row
        self.W.T[0] = 1

    def prediction(self, W, X_new):
        """
        Returns class predictions for a matrix of test data.
        Args:
        W (numpy array): trained matrix of weights
        X_new (predicted data): matrix of unseen weights
        """
        K = X_new.shape[0]
        pred_classes = np.zeros((K, W.shape[0])) - 1

        for k in range(K-1):
            pred_classes[k, :] = self._logistic(np.dot(W, X_new[k].T))
        return pred_classes.argmax(axis=1), pred_classes
