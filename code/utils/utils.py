import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import mne

import sys
sys.path.append('..')
from visualization_functions.signal_visualizers import temporal_plot, wavelets_plot, stft_plot


def load_Xy(path, verbose=0):
    ''' Load EEG signal and labels from .npz file. '''
    
    # Extract EEG and labels
    data = np.load(path)
    X = data['X']
    y = data['y']

    # Checking shapes
    if verbose:
        print('Loading ', path)
        print('X shape: {} - y shape: {}'.format(X.shape, y.shape))
        print('Labels: ', np.unique(y))
        for label in set(y):
            print('Label {} - {} trials'.format(label, len(np.where(y==label)[0])))
    return X, y

def inspect_mne(path, plot_mne=False, load=False, verbose=0, filetype='gdf'):
    assert filetype in ['gdf', 'vhdr'], "Please select a filetype among {'gdf', 'vhdr'}"
    
    # Load data as mne object
    print('Inspecting ', path)
    raw = mne.io.read_raw_gdf(path, preload=True) if filetype=='gdf' else mne.io.read_raw_brainvision(path, preload=True) 
    
    # Remove EOG channels
    raw.drop_channels(['Fp1', 'Fp2'])

    # Getting
    print('Sampling frequency: ', raw.info['sfreq'])
    
    # Getting ch_names
    print('Number of channels: ', len(raw.ch_names))
    print('Channel names: ', raw.ch_names)

    # Getting events
    events = mne.events_from_annotations(raw)[0]
    for label in np.unique(events[:,-1]):
        print("\t {}: {:.2f}".format(label, len(np.where(events[:,-1]==label)[0])))

    mapping = mne.events_from_annotations(raw)[1]
    for k,v in mapping.items():
        print(k, v)

    # Full info
    if verbose:
        print(raw.info)
        
    # Plot MNE signal
    if plot_mne:
        raw.plot(duration=5)
        
    # Return raw MNE object for further analysis
    if load:
        return raw
