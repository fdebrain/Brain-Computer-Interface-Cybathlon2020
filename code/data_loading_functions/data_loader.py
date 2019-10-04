import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocessing_functions.preproc_functions import rereference, filtering, standardize, clipping

class EEGDataset():
    def __init__(self, pilot_idx, data_path, fs=250, start=None, end=None,
                 labels_list=None, ch_list=None, 
                 rereferencing=False, standardization=False,
                 filt=False, f_type='butter', f_order=3, f_low=0, f_high=38,
                 valid_ratio=0., load_test = True, balanced=True):
        ''' Load dataset as numpy array of shape (n_trials, n_channels, n_samples).'''
        # Reproducibility
        np.random.seed(0)
        
        # Dataset
        self.data_path = data_path
        self.pilot_idx = pilot_idx
        
        # Loading parameters 
        self.labels_list = labels_list
        self.ch_list = ch_list
        self.balanced = balanced
        self.load_test = load_test
        self.valid_ratio = valid_ratio
        
        # Temporal parameters
        self.fs = fs
        self.start = start
        self.end = end
        
        # Pre-processing
        self.rereferencing = rereferencing
        self.standardization = standardization

        # Filtering
        self.filt = filt
        self.f_low = f_low
        self.f_high = f_high
        self.f_order = f_order
        self.f_type = f_type        
        
    def load_dataset(self):
        # Loading train data
        train_dataset = np.load(self.data_path + 'train/' + 'train{}.npz'.format(self.pilot_idx)) 
        X_train, y_train = train_dataset['X'], train_dataset['y']
        assert (len(X_train.shape)>2), "Loading train dataset failed !"

        self.n_channels = X_train.shape[1]
        self.ch_list = self.ch_list if self.ch_list else np.arange(self.n_channels)
        self.n_samples = X_train.shape[2]
        
        print('Properties: {} train trials - {} channels - {}s trial length'.format(X_train.shape[0], self.n_channels, self.n_samples/self.fs))
        
        # Loading test data
        if self.load_test:
            test_dataset = np.load(self.data_path + 'test/' + 'test{}.npz'.format(self.pilot_idx))
            X_test, y_test = test_dataset['X'], test_dataset['y']
            assert (self.n_channels == X_test.shape[1]), "Test dataset has incompatible n_channels."
            assert (self.n_samples == X_test.shape[2]), "Test dataset has incompatible n_samples."
        else:
            X_test = np.zeros(X_train.shape)
            y_test = np.zeros(y_train.shape)
        
        # Remap labels to {0,1,2,3,...}
        y_train = y_train - np.min(y_train)
        y_test = y_test - np.min(y_test)
        self.labels_list = self.labels_list if self.labels_list else list(set(y_train))
        
        # Selecting classes
        print('Selecting classes {} & balancing...'.format(self.labels_list))
        
        if self.balanced:
            n_balanced = np.min([sum(y_train == label) for label in self.labels_list])
            train_idx = np.concatenate([np.where(y_train == label)[0][:n_balanced] for label in self.labels_list])
            test_idx = np.concatenate([np.where(y_test == label)[0] for label in self.labels_list])
        else:            
            train_idx = np.concatenate([np.where(y_train == label)[0] for label in self.labels_list])
            test_idx = np.concatenate([np.where(y_test == label)[0] for label in self.labels_list])
        
        X_train = X_train[train_idx, :, :]
        y_train = y_train[train_idx]        
        X_test = X_test[test_idx, :, :]
        y_test = y_test[test_idx]
        assert (set(y_train) == set(self.labels_list)), "Available labels: {}".format(set(y_train))
        
        # Selecting channels 
        channels_idx = self.ch_list
        X_train = X_train[:, channels_idx, :]
        X_test = X_test[:, channels_idx, :]
        print('Selecting {} channels'.format(X_train.shape[1]))
        
        # Selecting time-window # Should be after preprocessing (especially filtering)
        fs = self.fs
        n_samples = self.n_samples
        start = 0 if self.start==None else int(self.start*self.fs)
        end = n_samples if self.end==None else int(self.end*self.fs)
        X_train = X_train[:, :, start:end]
        X_test = X_test[:, :, start:end]
        print('Selecting time-window [{} - {}]s - ({}s)...'.format(start/fs, end/fs, (end-start)/fs))

        # Preprocessing - Re-referencing (mean over channel axis should be 0 for each timestamp)
        if self.rereferencing:
            print("Re-referencing...")         
            X_train = rereference(X_train)
            X_test = rereference(X_test)

            # Sanity check
            assert np.isclose(np.mean(np.mean(np.mean(X_train, axis=1), axis=0)), 0)
            if self.load_test:
                assert np.isclose(np.mean(np.mean(np.mean(X_test, axis=1), axis=0)), 0)

        # Preprocessing - Filtering
        if self.filt:
            print("Filtering ({}-{})Hz...".format(self.f_low, self.f_high))
            X_train = filtering(X_train, self.fs, self.f_order, self.f_type, self.f_low, self.f_high)
            X_test  = filtering(X_test, self.fs, self.f_order, self.f_type, self.f_low, self.f_high)            
            
        # Preprocessing - Standardization
        if self.standardization:
            print("Clipping & standardizing...")
            X_train = clipping(X_train, 6)
            X_test = clipping(X_test, 6)
            X_train = standardize(X_train)
            X_test  = standardize(X_test)
            
            # Sanity check            
            assert np.isclose(np.mean(np.mean(np.mean(X_train, axis=-1), axis=0)), 0)
            assert np.isclose(np.mean(np.mean(np.std(X_train, axis=-1), axis=0)), 1, rtol=0.1)
            if self.load_test:
                assert np.isclose(np.mean(np.mean(np.mean(X_test, axis=-1), axis=0)), 0)
                assert np.isclose(np.mean(np.mean(np.std(X_test, axis=-1), axis=0)), 1, rtol=0.1)
        
        # Splitting into training & validation datasets
        if self.valid_ratio>0:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.valid_ratio, random_state=0, shuffle=True, stratify=y_train)
            assert np.isclose( [np.sum(y_train==c)/len(y_train) for c in np.unique(y_train)],
                               [np.sum(y_valid==c)/len(y_valid) for c in np.unique(y_valid)], atol=0.05).all()
        else:
            X_valid, y_valid = np.zeros(X_train.shape), np.zeros(y_train.shape)
            
        # Shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        
        if self.load_test==False:
            X_test, y_test = X_valid, y_valid
        
        print('Output shapes: ', X_train.shape, X_valid.shape, X_test.shape)
        print('Output classes: ', np.unique(y_train))
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def cropping(X, y, fs=250, n_crops=10, crop_len=2., shuffled=False):
    ''' Augment the number of trials by temporal cropping (with possible overlap).
        Inputs:
            - X, y: Input EEG signals and corresponding labels.
            - fs: Sampling frequency.
            - n_crops: Number of crops to extract for each trial.
            - crop_len: Length of each crop in seconds.
            - Shuffled: Allow additional shuffle.
        Outputs:
            - X_crops, y_crops: Output cropped EEG signals and corresponding labels.
            - trial2crops_idx: Input/output index mapping (which output cropped trial corresponds to which input trial).
    '''
    n_trials = len(y)
    if n_crops>1:
        # Cropping parameters
        crop_samples = int(crop_len * fs) # in samples
        crop_stride = int((X.shape[-1] - crop_samples) / (n_crops - 1)) # in samples
        overlap_ratio = 1 - crop_stride / crop_samples
        
        print('Cropping each trial into {} crops of {:.2}s with overlap ratio {:.2} ({} samples)'.format(n_crops, float(crop_len), overlap_ratio, crop_stride))
    
        X_crops = np.concatenate([ np.stack([ X[trial_idx, :, crop_idx*crop_stride : crop_idx*crop_stride + crop_samples] 
                                          for crop_idx in range(n_crops) ], axis=0)
                               for trial_idx in range(n_trials) ], axis=0)
    
        y_crops = np.concatenate([ [y[trial_idx]]*n_crops for trial_idx in range(y.shape[0]) ])
    else:
        #print('No cropping.')
        X_crops, y_crops = X, y

    # Get crops indexes corresponding to each trial
    trial2crops_idx = {}
    idx = np.arange(n_trials*n_crops)
    if shuffled:
            idx, X_crops, y_crops = shuffle(idx, X_crops, y_crops, random_state=0)
    for trial_idx in range(n_trials):    
        trial2crops_idx[trial_idx] = np.where(np.logical_and(idx>=trial_idx*n_crops, idx<(trial_idx+1)*n_crops))[0]
        
    return X_crops, y_crops, trial2crops_idx
