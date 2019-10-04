import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy
from tqdm import tqdm_notebook

class Morlet(object):
    def __init__(self, w0=6):
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        w = self.w0
        x = t / s
        output = np.exp(1j * w * x)
        if complete:
            output -= np.exp(-0.5 * (w ** 2))
        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)
        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** .5)

    def scale_from_period(self, period):
        """
        Compute the scale from the fourier period.
        Returns the scale
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4. * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -.25 * Hw * np.exp((-(x - self.w0) ** 2) / 2)    


class CWT1(TransformerMixin, BaseEstimator):
    def __init__(self, fs=250, dj=0.125, flatten=False):
        self.fs = fs
        self.dj = dj
        self.flatten = flatten
        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        n_trials, n_channels, _ = X.shape
        powers = np.array([ [self.compute_power(X[trial_idx, channel_idx, :]) for channel_idx in range(n_channels)]
                             for trial_idx in range(n_trials)])
        
        if self.flatten:
            powers = np.reshape(powers, (n_trials, -1))
        
        return powers
    
    def compute_power(self, X):
        wavelet = Morlet().frequency
        axis = -1
        
        # Time array
        dt = 1./self.fs
        n_samples = X.shape[axis]
        time = np.indices((len(X),)).squeeze() * dt
        
        # Set smallest resolvable scale
        def f(s):
            return Morlet().fourier_period(s) - 2 * dt
        s0 = scipy.optimize.fsolve(f, 1)[0]
        
        # Getting optimal scales
        dj = self.dj
        J = int((1 / dj) * np.log2(n_samples * dt / s0)) # Largest scale
        widths = s0 * 2 ** (dj * np.arange(0, J + 1))
        
        # Compute FFT data with padding
        pN = int(2 ** np.ceil(np.log2(n_samples)))
        fft_data = scipy.fft(X, n=pN, axis=axis)
        
        # Getting frequencies
        w_k = np.fft.fftfreq(pN, d=1./self.fs) * 2 * np.pi
        
        # Sample wavelets and normalize
        norm = (2 * np.pi * widths / dt) ** .5
        wavelet_data = norm[:, None] * wavelet(w_k, widths[:, None])
        
        # Perform the convolution in frequency space
        axis = (axis % X.ndim) + 1
        slices = [slice(None)] + [None for _ in X.shape]
        slices[axis] = slice(None)

        out = scipy.ifft(fft_data[None] * wavelet_data.conj()[slices],
                         n=pN, axis=axis)

        # Remove zero padding
        slices = [slice(None) for _ in out.shape]
        slices[axis] = slice(None, n_samples)
        
        if X.ndim == 1:
            out = out[slices].squeeze()
        else:
            out = out[slices]
        
        # Compute frequency power
        out = out[slices]
        power = (np.abs(out).T ** 2 / widths).T
        return power
