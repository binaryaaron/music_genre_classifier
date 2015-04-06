"""
bayes.py handles implementation of the meaty parts of the classifier -
estimation of priors and predicting new data.
"""
try:
    import numpy as np
    import utils
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")


__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


class LemmaTokenizer(object):
    """
    LemmaTokenizer is an optional tokenizer for the CountVectorizer. it
    provides stemming of words.
    This example is taken verbatim from Scikit-Learn's implementation.
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def phat_class_est(class_list, class_labels, debug=False):
    """calculates the priors of a class using MLE
    Args:
        class_list (np.array): the long list of classes for each document in
    the dataset
        class_labels(np.array): list of class label names
    Returns:
        np.array of prior counts for all
    """
    if debug:
        print('Estimating prior class probabilities')
    class_priors = np.zeros(shape=(len(class_labels)), dtype='float64')
    for i, lab in enumerate(class_labels):
        view = class_list[class_list == i]
        class_priors[i] = view.size/class_list.size
        if debug:
            print(lab + ': ' + str(class_priors[i]))
    return class_priors


def phat_word_est(train_data, labels, beta=None,
                  nclasses=20, laplacian=(False, None), debug=None):
    """Gets P(w|d) for all words in the dictionary.
    Args:
        train_data (scipy.sparse): training set, assumes bag of words
        labels(np.array): list of class label names (scikitlearn
            data.target_names) class_names(list): array of class labels for
            each document
        laplacian : tuple with bool and alpha specifying using it or not
        nclasses (int): number of classes in the set
    Returns: tuple with (numpy array of summed words, numpy array of priors)
    """
    vocab_n = train_data.shape[0]
    lap, alpha = laplacian
    if lap is True:
        if alpha is None:
            alpha = 1
        # this is done for easy of the for loop's calculation.
        _alpha = alpha+1
        denominator_p = alpha * nclasses
        print('estimating params with Laplacian Smoothing:')
        print('\t alpha = %d' % alpha)
        print('\t denominator = %d' % denominator_p)
    else:
        # beta is 1/(vocab size)
        if beta is None:
            beta = 1/vocab_n
        _alpha = 1 + beta
        denominator_p = beta * vocab_n
        print('estimating params with Dirchlet Prior:')
        print('\t vocabsize = %f' % vocab_n)
        print('\t beta = %f' % beta)
        print('\t alpha = %f' % _alpha)
        print('\t denominator = %f' % denominator_p)

    # gives the total number of words for a column vector in bow model
    word_sums = train_data.sum(axis=0)
    # preallocate array, 20 row, word dic length
    phat_words = np.zeros((nclasses, train_data.shape[1]), dtype='float64')

    # create sum vectors and assign them to the correct spot
    # may need to fiddle with smoothing params
    for i in range(nclasses):
        c_mask = labels == i
        c_sum = train_data[c_mask].sum(axis=0)
        c_sum = c_sum + (_alpha - 1)
        # print(c_sum/word_sums)
        phat_words[i] = (c_sum / (word_sums + denominator_p))
    # return (word_sums, phat_words)
    return phat_words


def predict(test_data, test_labels, p_classes, p_features, classes=20,
            debug=False):
    """Function to predict the class of a dataset.
    Args:
        test_data (scipy.sparse): This project assumes a bow model in a sparse
            matrix
        test_labels (numpy.array): label vector
        p_classes (numpy.array): 1-dim array of class priors.
        p_features (numpy.array): 2-d array of estimated feature probabilities.
            expected size is (classes, features)
        classes (int): number of classes
    Return:
        2-d numpy array with document ID, true label, and predicted label as
        columns. shape should be (number_test_documents, 3)
    """
    log_p_classes = np.log2(p_classes)
    log_p_features = np.log2(p_features)

    # adds log probs for each feature vector
    # print('log-prob array shape: ')
    # print(log_p_features.T.shape)

    # this gives us a (n_test_data, n_classes) matrix
    # corresponding to a document per column with a row of probabilities for
    # belonging to a class.
    # pred_log_probs = test_data.dot(sums)
    pred_log_probs = log_p_classes + (test_data * log_p_features.T)

    # gives vector of class labels for all test vectors
    # axis = 1 gives the argmax along each row
    # and conveniently gives a vector
    pred_labels = np.argmax(pred_log_probs, axis=1)
    acc = pred_labels == test_labels

    # returns (n, 3) array
    return np.array([test_labels, pred_labels, acc]).T


def vectorize(train_data, test_data, minfreq=5, maxfreq=0.90, stemmer=False,
              model='bow'):
    """Uses scikit-learn's tools to produce a bag of words model over the data.
    Args:
        train_data (list): list of documents in the training set
        test_data (list): list of documents in the test set
        minfreq: int or fraction of minimum documents that contain a word
        maxfreq: words that appear in more than this number are not used
        stemmer: specify a stemmer for the model e.g., LemmaTokenizer
        model: bag of word or tfidf.
    Return:
        tuple of two fitted bag-of-words models.
    """
    if model == 'bow':
        cv_train = CountVectorizer(stop_words='english',
                                   max_df=maxfreq,
                                   min_df=minfreq
                                   # analyzer='char_wb',
                                   # ngram_range=(2,2)
                                   # strip_accents='unicode'
                                   # token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
                                   # tokenizer=LemmaTokenizer()
                                   )
    elif model == 'tfidf':
        cv_train = TfidfVectorizer(stop_words='english',
                                   max_df=maxfreq,
                                   min_df=5
                                   # sublinear_tf=True
                                   )
    print('fitting training ' + model + ' vector model')
    train_ = cv_train.fit_transform(train_data)
    print('fitting test ' + model + ' vector model')
    test_ = cv_train.transform(test_data)
    return(train_, test_, cv_train)


def run_model(train_data, test_data, beta=None, bow=False, report=True,
              laplacian=(False, None)):
    """Runs the full training and testing steps
    Args:
        train_data(sklearn.datasets.base.Bunch): the training data from scikit
        test_data(sklearn.datasets.base.Bunch): the testing data from scikit
        beta: the value you want to use for beta
        bow: tuple with the bag of word models if you already have them
        report: flag for reporting
    Returns:
        tuple with all estimated things - (class_priors, estimated_words,
        predicted values, reportstring)
    """
    if bow is not False:
        return _run_model(train_data, test_data, beta, bow, report=report,
                          laplacian=laplacian)

    print("fitting count vectorizers")
    _bow = (tr_bow, tr_bow, cv) = vectorize(train_data.data, test_data.data)
    bow = (_bow[0], _bow[1])
    return _run_model(train_data, test_data, beta=beta, bow=bow, report=report,
                      laplacian=laplacian)


def _run_model(train_data, test_data, beta=None, bow=False, report=True,
               laplacian=(False, None)):
    """helper function for run_model
    Args:
        train_data(sklearn.datasets.base.Bunch): the training data from scikit
        test_data(sklearn.datasets.base.Bunch): the testing data from scikit
        beta: the value you want to use for beta
        bow: tuple with the bag of word models if you already have them
        report: flag for reporting
    Returns:
        tuple with all estimated things - (class_priors, estimated_words,
        predicted values, reportstring)
    """

    if bow is not False:
        train_bow, test_bow = bow
    print("estimating class priors")
    class_priors = phat_class_est(train_data.target,
                                  train_data.target_names,
                                  debug=False)

    print("estimating word priors")
    phat_words = phat_word_est(train_bow,
                               laplacian=laplacian,
                               beta=beta,
                               labels=train_data.target
                               )

    print("predicting")
    predicted = predict(test_data=test_bow,
                        test_labels=test_data.target,
                        p_classes=class_priors,
                        p_features=phat_words
                        )

    np.set_printoptions(precision=4)
    if report is True:
        rep = utils.report(predicted, train_data.target_names,
                           print_report=True,
                           print_cm=True)
    else:
        rep = utils.report(predicted, train_data.target_names,
                           print_report=False,
                           print_cm=False)

    return (class_priors, phat_words, predicted, rep)


def get_newsgroups(remove=[]):
    """Convenience function to get the newgroups dataset.
    Args:
        remove (list): items to filter from the data, headers, quotes, or
        footers. defaults to nothing
    Returns:
        tuple of both scikit training / testing sets.
    """
    print('loading training set')
    twenty_train = fetch_20newsgroups(subset='train',
                                      remove=remove,
                                      shuffle=False)
    print('loading testing set')
    twenty_test = fetch_20newsgroups(subset='test',
                                     remove=remove,
                                     shuffle=False)
    return (twenty_train, twenty_test)


def main():
    twenty_train, twenty_test = get_newsgroups()
    cpriors, map_, predicted, report = run_model(twenty_train, twenty_test)

if __name__ == "__main__":
    main()
