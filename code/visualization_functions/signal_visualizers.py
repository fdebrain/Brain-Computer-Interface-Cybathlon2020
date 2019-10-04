import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal
from ipywidgets import interact


def temporal_plot(X, fs=250, title='Amplitude', fig=None, ax_idx=0):
    ''' Plot signal in temporal space (amplitude vs time).
    Input:
        - X: Signal to plot (numpy array of shape (n_samples,)).
        - fs: Sampling frequency in Hz (int).
        - title: Plot title (string).
        - fig: External figure (object).
        - ax_idx: External axis index (int).
    Output:
        - 2D plot (amplitude vs time).
    '''
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
    
    assert len(X.shape) == 1, "X should be of shape (n_samples,)."
    ax.plot(X)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    tick_locs = np.arange(0, X.shape[-1] + fs, fs)
    tick_lbls = np.arange(X.shape[-1]//fs + 2)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_lbls)
    return fig

def amplitude2D_plot(X, fs=250, y_label='Trials', vmax=10, title='Voltage amplitude', fig=None, ax_idx=0):
    ''' Plot signal amplitude trial- or channel-wise in temporal space (trials/channels vs time).
    Input:
        - X: Signal to plot (numpy array of shape (n_trials/n_channels, n_samples)).
        - fs: Sampling frequency (int).
        - channel_idx: Electrode selection if trial-wise plot (int).
        - y_label: Label of y-axis (string).
        - vmax = Amplitude display threshold (int).
        - title: Plot title (string).
        - fig: External figure (object).
        - ax_idx: External axis index (int).
    Output:
        - 2D color plot (trials vs time).
    '''
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
    
    assert y_label in ['Trials', 'Channels'], "Please choose y_label among {'Trials', 'Channels'}."

    if y_label=='Trials':
        assert len(X.shape) == 2, "X should be of shape (n_trials, n_samples)."
        ax.imshow(X, aspect='auto', cmap='magma', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_ylabel('Trials')

    else:
        assert len(X.shape) == 2, "X should be of shape (n_channels, n_samples)."
        ax.imshow(X, aspect='auto', cmap='magma', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_ylabel('Channels')        

    ax.set_xlabel('Time [s]')
    tick_locs = np.arange(0, X.shape[-1], fs)
    tick_lbls = np.arange(X.shape[-1]//fs + 1)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_lbls)
    ax.set_title(title)
    return fig

def psd_plot(X, fs=250, nperseg=250, title='PSD', fig=None, ax_idx=0):
    ''' Plot power spectral frequency (PSD vs frequency).
    Input:
        - X: Signal to plot (numpy array of shape (n_samples,)).
        - fs: Sampling frequency in Hz (int).
        - npersef:  Number of samples per FFT segment (int).
        - title: Plot title (string).
        - fig: External figure (object).
        - ax_idx: External axis index (int).
    Output:
        - 2D plot (PSD vs frequency).
    '''
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
        
    assert len(X.shape) == 1, "X should be of shape (n_samples,)."
    f, Pxx_den = scipy.signal.welch(X, fs, nperseg=nperseg)
    ax.semilogy(f, Pxx_den)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [V^2/Hz]')
    ax.set_title(title)
    return fig

def stft_plot(X, fs=250, f_min=None, f_max=None, nperseg=70, r_nover=0.99, r_nfft=8, title='STFT', fig=None, ax_idx=0):
    ''' Plot signal in frequency space (frequency vs time) to visualize the change of a 
    nonstationary signal's frequency content over time.
    Input:
        - X: Signal to plot (numpy array of shape (n_samples,)).
        - fs: Sampling frequency in Hz (int).
        - nperseg: Number of samples per FFT segment (int).
        - r_nover: n_overlap ratio (float<1).
        - r_nfft: NFFT ratio (float>1).
        - log: (boolean).
        - title: Plot title (string).
        - fig: External figure (object).
        - ax_idx: External axis index (int).
    Output:
        - 2D color plot (frequency vs time).
    '''
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()

    assert len(X.shape) == 1, "X should be of shape (n_samples,)."
    f, t, Sxx = scipy.signal.spectrogram(X, fs, nperseg=nperseg, nfft=r_nfft*nperseg, noverlap=r_nover*nperseg)
    #f, t, Sxx = scipy.signal.stft(X, fs, nperseg=nperseg, nfft=r_nfft*nperseg, noverlap=r_nover*nperseg)

    ax.pcolormesh(t, f, np.abs(Sxx), cmap='magma')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_ylim([f_min, f_max])
    ax.set_title(title)
    return fig

''' Source: https://github.com/obspy/obspy/blob/06906f389bdaeb5fa4c4bd0cab9f066082e7da42/obspy/signal/tf_misfit.py '''
def wavelets_plot(X, fs=250, n_freqs=100, f_min=1, f_max=40, w0=8, log_scale=False, title='CWT', fig=None, ax_idx=0):
    ''' Apply continuous wavelets transform to visualize the change of a 
    nonstationary signal's frequency content over time.
    Input:
        - X: Signal to plot (numpy array of shape (n_samples,)).
        - fs: Sampling frequency in Hz (int).
        - n_freqs: Number of wavelets (int).
        - f_min, f_max: Max/min frequency on y-axis (int>0).
        - w0: Morlet wavelets parameter (int).
        - log: Scale of y-axis (boolean).
        - title: Plot title (string).
        - fig: External figure (object).
        - ax_idx: External axis index (int).
    Output:
        - 2D color plot (frequency vs time).
    '''    
    
    dt = 1./fs
    npts = X.shape[-1] * 2
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)

    cwt = np.zeros((npts // 2, n_freqs), dtype=np.complex)

    # Morlet wavelet
    def psi(t):
        return np.pi ** (-.25) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2.)

    def scale(f):
        return w0 / (2 * np.pi * f)
    
    buf = math.ceil(math.log(npts) / math.log(2))
    next_pow = int(math.pow(2, buf))
    nfft = next_pow * 2
    sf = np.fft.fft(X, n=nfft)

    # Ignore underflows.
    with np.errstate(under="ignore"):
        for n, _f in enumerate(f):
            a = scale(_f)
            # time shift necessary, because wavelet is defined around t = 0
            psih = psi(-1 * (t - t[-1] / 2.) / a).conjugate() / np.abs(a) ** .5
            psihf = np.fft.fft(psih, n=nfft)
            tminin = int(t[-1] / 2. / (t[1] - t[0]))
            cwt[:, n] = np.fft.ifft(psihf * sf)[tminin:tminin + npts // 2] * (t[1] - t[0])
            
    t = np.linspace(0, X.shape[-1]/fs, X.shape[-1])
    x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), cwt.shape[1]))
        
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
    
    ax.pcolormesh(x, y, np.abs(cwt.T), cmap='magma');
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    return fig

def tsne_plot(X, y, perplexity=20, title='t-SNE', label_idxs=[0,1,2,3], label_names=['Right', 'Left', 'Tongue', 'Feet'], fig=None, ax_idx=0):
    ''' Visualize the EEG trials in optimized 2D embedding space.
    Input:
        - X: EEG trials (numpy array of shape (n_trials, n_channels, n_samples)).
        - y: Labels (numpy array of shape (n_trials,)).
        - perplexity: Hyperparameter for t-SNE (int).
        - embedding_name: Name of the embedding (string).
    Output:
        - 2D scatter plot (feature 1 vs feature 2).
    '''
    from sklearn.manifold import t_sne
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
    
    n_trials = X.shape[0]
    out = t_sne.TSNE(n_components=2, perplexity=perplexity, n_iter=500, random_state=0).fit_transform(X.reshape((n_trials,-1)))
    outs = np.array([ out[y==c] for c in label_idxs ])
    
    colors = np.array(['b','r','g','orange'])
    [ ax.scatter(*outs[i][:,:].T, c=colors[l], alpha=0.7) for i,l in enumerate(label_idxs) ]
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend([ label_names[l] for l in label_idxs ], loc=0)
    ax.set_title(title)
    return fig

def activation_plot(X, model, layer_idx, title='Activation'):
    print(model.layers[layer_idx])
    get_layer_outputs = K.function([model.layers[0].input], [model.layers[layer_idx].output])
    output = np.squeeze(get_layer_outputs([X])[0])
    frequency_plot(X_train[0,0,depth_idx], NFFT=250, log=False, f_max=50)

def visual_analysis(X, fs=250, f_min=4, f_max=30, n_perseg=125, n_freqs_wav=500, w0_wav=16, log_scale_wav=False, figsize=(9,3)):
    fig, ax = plt.subplots(1,3, figsize=figsize)
    fig.tight_layout()
    
    @interact(trial_idx=(0, X.shape[0]-1), channel_idx=(0, X.shape[1]-1))
    def plot(trial_idx=0, channel_idx=0):
        [ ax[i].clear() for i in [0,1,2]]
        temporal_plot(X[trial_idx, channel_idx,:], fs, fig=fig, ax_idx=0);
        stft_plot(X[trial_idx, channel_idx,:], fs, f_min, f_max, n_perseg, fig=fig, ax_idx=1);
        wavelets_plot(X[trial_idx, channel_idx,:], fs, n_freqs_wav, f_min, f_max, w0_wav, 
                      log_scale_wav, fig=fig, ax_idx=2);