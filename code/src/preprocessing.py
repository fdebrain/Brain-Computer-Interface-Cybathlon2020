import numpy as np
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin


def preprocessing(signal, fs, rereference=False, filt=False, standardize=False):
    if rereference:
        signal = rereferencing(signal)
    if filt:
        signal = filtering(signal, fs, f_order=2,
                           f_low=4, f_high=38)
    if standardize:
        signal = clipping(signal, sigma=6)
        signal = standardizing(signal)
    return signal


def get_CAR():
    search_space = {}
    preproc = CommonAverageReference()
    return preproc, search_space


def get_CN(sigma):
    search_space = {}
    preproc = ClipNormalizer(sigma)
    return preproc, search_space


def get_BPF(fs, f_order, f_type, f_low, f_high):
    search_space = {'bpf__f_order': (2, 5),
                    'bpf__f_low': [0, 2, 4],
                    'bpf__f_high': [30, 40, 100]}
    preproc = BandPassFilter(fs, f_order, f_type, f_low, f_high)
    return preproc, search_space


def get_preprocessor(selected_preproc, config):
    """[summary]

    Arguments:
        selected_preproc {List[str]} -- List of preprocessing step names
        config {dict} --

    Returns:
        List[tuples], List[int] -- List of (preproc_name, object).
    """
    preproc_steps = []
    search_space = {}
    for preproc in selected_preproc:
        if preproc == 'CAR':
            car, _ = get_CAR()
            preproc_steps.append(('car', car))

        if preproc == 'CN':
            cn, ss = get_CN(**config['CN'])
            preproc_steps.append(('cn', cn))
            search_space = {**search_space, **ss}

        if preproc == 'BPF':
            bpf, ss = get_BPF(**config['BPF'])
            preproc_steps.append(('bpf', bpf))
            search_space = {**search_space, **ss}

    return preproc_steps, search_space


class CommonAverageReference(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return rereferencing(X)


class ClipNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = clipping(X, self.sigma)
        return standardizing(X)


class BandPassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, f_order, f_type, f_low, f_high):
        self.fs = fs
        self.f_order = f_order
        self.f_type = f_type
        self.f_low = f_low
        self.f_high = f_high

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return filtering(X, self.fs, self.f_order, self.f_type,
                         self.f_low, self.f_high)


def rereferencing(X):
    ''' Apply Common Average Reference to the signal. At each timestep the sum of all channel values should be zero. '''
    average_channels = np.mean(X, axis=-2, keepdims=True)
    return X - average_channels


def standardizing(X, eps=1e-8):
    ''' Outputs the standardized signal (zero mean, unit variance).'''
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    return (X-mean)/(std+eps)


def perturbate(X, sigma):
    ''' Add noise to input signal.'''
    noise = sigma * np.random.randn(*X.shape)
    return X + noise


def filtering(X, fs=250, f_order=5, f_type='butter', f_low=4, f_high=38):
    ''' Apply filtering operation on the input data using Second-order sections (sos) representation of the IIR filter (to avoid numerical instabilities).'''

    filt_params = {'N': f_order,
                   'output': 'sos',
                   'fs': fs}

    if f_type == 'butter':
        filt = scipy.signal.butter
    elif f_type == 'cheby':
        filt = scipy.signal.cheby2
        filt_params['rs'] = 40
    elif f_type == 'ellip':
        filt = scipy.signal.ellip
        filt_params['rs'] = 40
        filt_params['rp'] = 5
    else:
        raise ValueError(
            "Please chose f_type among {'butter', 'cheby', 'ellip'}.")

    if f_low == 0:
        filt_params['Wn'] = [f_high]
        sos = filt(**filt_params, btype='lowpass')
    elif f_high == 0:
        filt_params['Wn'] = [f_low]
        sos = filt(**filt_params, btype='highpass')
    else:
        filt_params['Wn'] = [f_low, f_high]
        sos = filt(**filt_params, btype='bandpass')

    X_bandpassed = scipy.signal.sosfilt(sos, X)
    return X_bandpassed


def clipping(X, sigma):
    ''' Outputs clipped signal by setting min/max boundary amplitude values (+-sigma*std).'''
    median = np.median(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)

    # Clipping boundaries
    tops = median + sigma*std
    bottoms = median - sigma*std
    X_clipped = X.copy()

    if len(X.shape) == 2:
        n_channels, n_samples = X.shape
        for channel_idx in range(n_channels):
            top = tops[channel_idx]
            bottom = bottoms[channel_idx]
            X_clipped[channel_idx, X[channel_idx, :] > top] = top
            X_clipped[channel_idx, X[channel_idx, :] < bottom] = bottom

    elif len(X.shape) == 3:
        n_trials, n_channels, n_samples = X.shape

        for trial_idx in range(n_trials):
            for channel_idx in range(n_channels):
                top = tops[trial_idx, channel_idx]
                bottom = bottoms[trial_idx, channel_idx]
                X_clipped[trial_idx, channel_idx,
                          X[trial_idx, channel_idx, :] > top] = top
                X_clipped[trial_idx, channel_idx,
                          X[trial_idx, channel_idx, :] < bottom] = bottom
    return X_clipped


def crop_single_trial(x, stride, n_samples, n_crops):
    """Crop a single trial into n_crops of size crop_len.

    Arguments:
        x {np.array} -- EEG data of shape (n_channels, n_samples)

    Keyword Arguments:
        stride {int} -- Interval between starts of two consecutive crops in samples.
        n_samples {int} -- Time sample for each output crop.
        n_crops {int} -- Number of desired output crops.
    """
    assert stride > 0, 'Stride should be positive !'
    X_crops = np.stack([x[:, i*stride: i*stride + n_samples]
                        for i in range(n_crops)], axis=0)
    return X_crops


def cropping(X, y, fs=500, n_crops=50, crop_len=0.5):
    assert n_crops > 1, 'Use n_crops > 1 !'

    # Cropping parameters
    n_samples = int(crop_len * fs)  # samples
    stride = int((X.shape[-1] - n_samples) / (n_crops - 1))  # samples
    overlap_ratio = 1 - stride / n_samples

    X_crops = np.concatenate([crop_single_trial(x, stride, n_samples, n_crops)
                              for x in X], axis=0)
    y_crops = np.concatenate([[y[trial_idx]] * n_crops
                              for trial_idx in range(len(y))])

    return X_crops, y_crops
