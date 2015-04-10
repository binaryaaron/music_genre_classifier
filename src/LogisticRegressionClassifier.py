"""
LogisticRegressionClassifer.py handles implementation of the meaty parts of the classifier -
estimation of priors and predicting new data.
"""


__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


try:
    import numpy as np
    from scipy.special import expit
    from sklearn.cross_validation import KFold
    import utils 


except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")


class LogisticRegressionClassifier(object):
    """
    Main class for the classifer.
    """

    def __init__(self, X, y, class_labels, eta=0.0001, _lambda=0.001):
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
        self.X_test = None
        self.y_test = None
        self.class_dict = class_labels

        # ######### Parameters needed by log.regression
        self.Delta = None
        self.W = None
        self.eta = eta
        self._lambda = _lambda
        self.P = None

        self.fitted_model = None

    def make_Delta(self):
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

    def sig_prime(self, x):
        """
        First derivative of the sigmoid function.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def _logsumexp(self, X, Y):
        """
        handy calculator for the log sum exp trick
        """
        return np.log(np.exp(np.dot(X, Y)))

    def P(self, X, W):
        """
        Calculates the logistic transform (i.e., probabilities) used in
        training the model.
        Args:
            X (np.array) 2-d numpy array of features
            W (np.array) 2d numpy array of weights
        """
        # number of classes
        K = W.shape[0]
        P = np.zeros(shape=(K, X.shape[0]))
        P = np.dot(W, X.T)
        P[K-1] = 1
        colsums = np.sum(P, axis=0)
        P = P/colsums
        self.P = np.exp(P)
        return self.P

    def grad_decent(self, rounds):
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

        _lambda = self._lambda
        error = 0

        for i in range(rounds):
            d = np.dot(self.Delta - np.log(self.P(self.X, self.W)), self.X)
            # gets the largest deviation
            _error = np.abs(d).max()
            if _error > error:
                # update learning rate
                self.eta /= (1 + i % 3/rounds)
                print('Step %d: Error: %f \n \
                      \t updating learning rate: %f' % (i, error, self.eta))
            error = _error
            # f = d - (_lambda * W)
            # W += (eta * f)
            self.W += (self.eta * (d - (_lambda * self.W)))

        print('Error: %f \n Learn rate: %f' % (error, self.eta))
        # return W

    def cross_validate(self):
        kf = KFold(self.X.shape[0], n_folds=2)
        for train, test in kf:
            X_train, X_test, y_train, y_test = \
                self.X[train], self.X[test], self.y[train], self.y[test]

    def train(self, rounds=1000, cv=False):
        """
        Trains the model on the currently held data.
        """

        if self.W is None:
            print('initializing weight matrix for the first time')
            self.init_weights()

        if cv is False:
            self.grad_decent(rounds=1000)
        else:
            self.cross_validate()
        return

    def init_weights(self):
        self.W = 0 + np.zeros(shape=(len(self.class_labels),
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
            pred_classes[k,:] = _logistic(np.dot(W, X_new[k].T))
        return pred_classes.argmax(axis=1), pred_classes


    def plot_confusion_matrix(cm, title='Confusion matrix', normalized=True,
                                cmap=plt.cm.Oranges, save_file=""):
        """
        Displays the confussion matrix indicated by `cm`. If argument
        `normalized` is Ture, then the matrix is normalized. Optionally
        the image can be saved to a file

        Arguments:
        ----------
        `cm`: The confusion matrix to be displayed.
        `title`: The title for the window.
        `normalized`: If True, normalizes the matrix before showing it.
        `cmap`: Colormap to use.
        `save_file`: If string different than empty, the resulting image is
        stored in such file.
        """
        if normalized:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save_file:
            plt.savefig(save_file)

        return cm

