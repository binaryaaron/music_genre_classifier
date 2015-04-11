#!/bin/python

try:
    import features as feat
    # import scipy.io.wavfile as wav
    import numpy as np
    # needed for the recursive glob
    import glob2
    from scipy.io.wavfile import read
    # from scikits.talkbox.features import mfcc
except ImportError:
    raise ImportError("""This program requires glob2, matplotlib, numpy, scipy,
    and sklearn, all of which can be installed via \n
    `pip3 install requirements.txt` in the root directory of this repo.""")


def _extract_mfcc(filename):
    """Extracts mfccs from wav files"""
    savename = filename[0: len(filename)-4] + '.mfc'
    samp_rate, X = read(filename)
    # ceps, mspec, spec = mfcc(X)
    ceps = feat.mfcc(X, samp_rate)
    num_ceps = ceps.shape[0]
    x = np.mean(ceps[int(num_ceps * 1/10):int(num_ceps * 9/10)], axis=0)
    np.save(savename, x)


def extract_feats(path='/cs/genres/*/*.wav', feature='mfcc'):
    """Extracts features from a  """
    classes = [line for line in glob2.glob(path)]
    for line in classes:
        print('extracting mfc from %s' % line)
        if feature == 'mfcc':
            _extract_mfcc(line)
