"""
Fun with logisitc regresion
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
    from sklearn.cross_validation  import StratifiedKFold
    import sklearn.metrics as metrics
    from LogisticRegressionClassifier import LogisticRegressionClassifier
    if sys.version_info < (3, 2):
        # python too old, kill the script
        sys.exit("This script requires Python 3.2 or newer! exiting.")
except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")

def main():

    # read data
    fft_dict, fft_labels, ffts = utils.read_features(feature='fft')
    mfc_dict, mfc_labels, mfcs = utils.read_features(feature='mfc')
    # fit classifiers
    lrc_fft = LogisticRegressionClassifier(ffts, fft_labels, fft_dict)
    lrc_mfc = LogisticRegressionClassifier(mfcs, mfc_labels, mfc_dict)
    # cross validate
    lrc_fft.cross_validate(k=10)
    # get high variance features and retrain
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(0.01150)
    fft_variance = sel.fit_transform(ffts)
    lr_fftvar = LogisticRegressionClassifier(a, fft_labels, fft_dict)
    lr.fftvar.cross_validate()

    # try with fscores
    f = feature_selection.SelectPercentile(f_classif(ffts, fft_labels),
            percentile=20)
    high_fscore = ffts[:,np.argsort(fscores)[::-1][1:][:200]]
    high_fscore.shape

    lr = LogisticRegressionClassifier(high_fscore, fft_labels, fft_dict)
    lr.cross_validate(3)



