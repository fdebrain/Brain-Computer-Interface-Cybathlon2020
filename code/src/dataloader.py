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
