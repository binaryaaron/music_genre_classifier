try:
    eval("print x")
except:
    raise FutureWarning
    import sys
    sys.exit()
try:
    import numpy as np
    import matplotlib.pyplot as plt
    # needed for the recursive glob
    import glob2
    import glob
    from scipy.io.wavfile import read
    import scipy as sp
except ImportError:
    raise ImportError("""This program requires glob2, matplotlib, numpy, scipy,
    and sklearn, all of which can be installed via \n
    `pip3 install requirements.txt` in the root directory of this repo.""")


def _extract_fft(filename):
    """Extracts the ffyfeatures from a file and saves them
        in numpy binary format"""
    savename = filename[0: len(filename)-4] + '.fft'
    samp_rate, X = read(filename)
    a = sp.fft(X, n=1000)
    np.save(savename, a)


def _extract_mcc(filename):
    """Extracts """
    savename = filename[0: len(filename)-4] + '.mc'
    samp_rate, X = read(filename)
    a = sp.fft(X, n=1000)
    np.save(savename, a)


def extract_feats(path='/cs/genres/*/*.wav', feature='fft'):
    """Extracts features from a  """
    classes = [line for line in glob2.glob(path)]
    for line in classes:
        print('extracting fft from %s' % line)
        if feature == 'fft':
            _extract_fft(line)
        if feature == 'mcc':
            _extract_mcc(line)
