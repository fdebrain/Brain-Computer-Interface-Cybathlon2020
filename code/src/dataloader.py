import logging
from collections import Counter

import numpy as np


def load_session(session_path, start=None, end=None):
    logging.info(f'Loading {session_path}')
    data = np.load(session_path)
    X, y, fs, ch_names = data['X'], data['y'], data['fs'], data['ch_names']

    logging.info(f'Select time-window {start}s to {end}s')
    start = 0 if start is None else int(start * fs)
    end = X.shape[-1] if end is None else int(end * fs)
    X = X[:, :, start:end]

    logging.info(f'Output data shape: X {X.shape} - y {y.shape}')
    logging.info(f'Available classes: {Counter(y)}')
    logging.info(f'Sample frequency: {fs}Hz')

    return X, y, int(fs), ch_names


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

    # logging.info(f'Cropping each trial into {n_crops} crops of '
    #              f'{float(crop_len):.2}s with overlap ratio '
    #              f'{overlap_ratio:.2} (stride: {stride} samples)')

    X_crops = np.concatenate([crop_single_trial(x, stride, n_samples, n_crops)
                              for x in X], axis=0)
    y_crops = np.concatenate([[y[trial_idx]] * n_crops
                              for trial_idx in range(len(y))])

    return X_crops, y_crops
