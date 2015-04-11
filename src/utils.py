"""
Utils provides the utilties for the classifer and whatnot.
"""

__author__ = "Aaron Gonzales"
__copyright__ = "MIT"
__license__ = "MIT"
__email__ = "agonzales@cs.unm.edu"


try:
    import numpy as np
    import matplotlib.pyplot as plt
    # needed for the recursive glob
    import glob2
    import glob
    from scipy.io.wavfile import read
    import scipy as sp
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing as pre
except ImportError:
    raise ImportError("""This program requires glob2, matplotlib, numpy, scipy,
    and sklearn, all of which can be installed via 
    `pip3 install requirements.txt`
    in the root directory of this repo.""")

def read_features(directory='../data/', feature='fft'):
    """
    Utilty function to read all the features and labels into a numpy
        array
    Args:
        directory (string) : path to the data dir
        feature (string): the feature you want to get
    Returns:
        Tuple of (class labels, featureset)
    """
    def filter_ffts():
        features = np.array([np.load(f) for f in all_features])
        # take the absolute value of the ffts
        features = np.absolute(features)
        # scaler = MinMaxScaler(feature_range=(0,1))
        # features = scaler.fit_transform(features)
        # normalizes over each feature
        features = features / np.max(features, axis=0)
        # inserts the proper row of ones to the matrix
        features = np.insert(features, 0, 1, axis=1)
        return features

    def filter_mfc():
        mfcs =  np.array([np.load(f) for f in all_features])
        # mfcs = mfcs / np.max(mfcs, axis=0)
        # scaler = MinMaxScaler(feature_range=(0,1))
        # mfcs = scaler.fit_transform(mfcs)
        pre.scale(mfcs, copy=False)
        return np.insert(mfcs, 0, 1, axis=1)

    glob_id = directory + '*.' + feature + '.npy'
    all_features = glob.glob(glob_id)
    # trims filename and splits on periods to get just the class type
    class_name_list = [name[8:len(name)-1].split('.')[0] for name in
                       all_features]
    # makes a set
    classes = {name for name in class_name_list}
    # makes a dict of features with numeric label
    classes = {name: val for val, name in enumerate(classes)}
    # makes a vector of labels
    label_vec = [classes[lab] for lab in class_name_list]
    class_labels = np.array(label_vec)
    # reads all the features in via numpy
    if feature == 'fft':
        return classes, class_labels, filter_ffts()
    return (classes, class_labels, filter_mfc())


def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          normalized=True,
                          cmap=plt.cm.Oranges,
                          save_file=""):
    """
    Displays the confusion matrix indicated by `cm`. If argument
    `normalized` is True, then the matrix is normalized. Optionally
    the image can be saved to a file. This was taken from Andres and I's
    version from the malware project

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


def extract_ffts():
    """
    Extracts the first 1000 ffts from the list of wavfiles.
    """

    def _extract_feats(filename):
        savename = filename[0: len(filename)-4] + '.fft'
        samp_rate, X = read(filename)
        ffts = sp.fft(X, n=1000)
        np.save(savename, ffts)

    path = '/cs/genres/*/*.wav'
    classes = [line for line in glob2.glob(path)]
    for line in classes:
        print("extracting %s" % line)
        _extract_feats(line)
