"""
Fun with logisitc regresion
"""

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


try:
    import numpy as np
    import sklearn.feature_selection as feature_selection
    from LogisticRegressionClassifier import LogisticRegressionClassifier
    import utils as utils
    import sys
    if sys.version_info < (3, 2):
        # python too old, kill the script
        sys.exit("This script requires Python 3.2 or newer! exiting.")
except ImportError:
    raise ImportError("This program requires Numpy and scikit-learn")


def main():
    print("We're going to do some regression now")
    # read data
    print("Reading the data from ../data/ ; run me from /src")
    fft_dict, fft_labels, ffts = utils.read_features(feature='fft')
    mfc_dict, mfc_labels, mfcs = utils.read_features(feature='mfc')
    # fit classifiers
    lrc_fft = LogisticRegressionClassifier(ffts, fft_labels, fft_dict)
    lrc_mfc = LogisticRegressionClassifier(mfcs, mfc_labels, mfc_dict)
    # cross validate
    print("training the first fft model with 10-fold CV")
    lrc_fft.cross_validate(k=10)
    # get high variance features and retrain
    print("extracting features from that model/the fft data")
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(0.01150)
    fft_variance = sel.fit_transform(ffts)
    lr_fftvar = LogisticRegressionClassifier(fft_variance, fft_labels, fft_dict)
    print("training the first reduced fft model with 10-fold CV")
    lr.fftvar.cross_validate()

    # try with fscores
    print("training the mfcs with 10-fold CV")
    lrc_mfc.cross_validate(10)

if __name__ == "__main__":
    main()
