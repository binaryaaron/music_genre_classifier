import scipy as sp

fn = 'filename'

sample_rate X = sp.io.wavfile.read(fn)
fft_features = abs(sp.fft(X)[:1000])
