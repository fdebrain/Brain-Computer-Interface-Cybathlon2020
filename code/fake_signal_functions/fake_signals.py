import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_transcient(f=5, mu=3.5, sigma=1, length=7, fs=250):
    N = int(length*fs)
    t = np.linspace(0, length, N)
    y = np.cos(2*np.pi*f*t)*np.exp(-(t-mu)**2/(2*sigma**2))
    return t,y

def create_sin_wave(f=5, length=7, fs=250):
    N = int(length*fs)
    t = np.linspace(0, length, N)
    y = np.sin(2*np.pi*f*t)
    return t,y

def create_chirp(f_min=5 ,f_max=10, length=7, fs=250):
    N = int(length*fs)
    t = np.linspace(0, length, N)

    # Frequencies with Laplace distribution 
    fm = np.exp(-np.abs(t - np.mean(t)))
    fm = fm - np.min(fm)
    fm = fm / np.max(fm)
    fm = fm*(f_max-f_min) + f_min  # Rescale
    
    # Create chirp
    chirp = np.sin(2*np.pi*(t + np.cumsum(fm)/fs))
    return t, chirp, fm

def fake_EEG(f_low=5, f_high=50, n_channels=10, length=7):
    freqs=np.logspace(np.log10(f_low), np.log10(f_high), n_channels)
    X = np.vstack([ create_gaussian_transcient(f=freqs[i], 
                                               mu=np.random.randn() + 3.5, 
                                               sigma=np.random.rand(),
                                               length=length)[1] for i in range(len(freqs))])
    return X

def fake_EEG_dataset(n_trials_per_label=4, f_low=5, f_high=50, n_channels=10, length=7, labels=[0,1,2,3]):
    X = np.stack([fake_EEG(f_low, f_high, n_channels, length) for _ in range(n_trials_per_label*len(labels))], axis=0)
    y = np.array([ [labels[i]]*n_trials_per_label for i in range(len(labels)) ]).flatten()
    return X, y

def plot_signal(t, y, title='', ylabel='Amplitude'):
    plt.figure()
    plt.plot(t,y)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    return plt