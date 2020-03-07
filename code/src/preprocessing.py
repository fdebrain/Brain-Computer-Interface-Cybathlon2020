import numpy as np
import scipy.signal


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
    mean = np.mean(X, axis=-1, keepdims=True)
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
