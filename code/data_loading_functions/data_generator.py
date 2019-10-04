import numpy as np
import scipy.signal
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocessing_functions.preproc_functions import rereference, filtering, standardize, clipping

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, batch_size=32, class_idxs=[0,1,2,3], class_nb_idx=[120,40,40,40], shuffle=True, preprocessors=[]):
        'Initialization'
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Get dataset properties
        data = np.load(self.data_path + '0.npz')['X']
        self.n_channels, self.n_samples = data.shape
        self.n_trials = len(glob.glob(self.data_path + '*.npz')) - 1
        self.class_idxs = class_idxs
        self.class_nb_idx = class_nb_idx
        self.n_classes = len(class_idxs)
        
        self.total_samples_per_class = np.min(class_nb_idx)
        self.samples_per_class = self.batch_size//self.n_classes
        
        assert(self.batch_size % self.n_classes == 0) #, 'Batch size need to be a multiple of the number of classes.')
        assert(self.samples_per_class <= self.total_samples_per_class) #, 'Reduce the batch size.')
        
        # Get all IDs (data filenames) of given classes 
        self.IDs = np.arange(self.n_trials)
        
        # Preprocessing
        self.preprocessors = preprocessors
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch aka n_samples_per_class / batch_size'
        return int((self.n_trials/4)*(self.n_classes/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = np.array([], dtype=int)
        for class_idx in self.class_idxs:
            indexes = np.append(indexes, self.indexes[class_idx][index*self.samples_per_class : (index+1)*self.samples_per_class])
        list_IDs_temp = [self.IDs[k] for k in indexes.ravel()]
            
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        cumul_idxs = np.array([np.sum(self.class_nb_idx[:i]) for i in range(len(self.class_nb_idx) + 1)], dtype=int)
        self.indexes = np.array([np.arange(cumul_idxs[i], cumul_idxs[i+1]) for i in range(len(cumul_idxs) - 1)])
        
        if self.shuffle:
            [np.random.shuffle(self.indexes[c]) for c in self.class_idxs]            

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, self.n_samples))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, trial_idx in enumerate(list_IDs_temp):
            # Store sample
            data = np.load(self.data_path + '{}.npz'.format(trial_idx))
            X[i,] = data['X']
            y[i] = data['y']
            
            # Preprocessing
            if "filter" in self.preprocessors:
                #print("Filtering...")
                X[i,] = filtering(X[i,], fs=250, f_order=3, f_low=4, f_high=38)
            if "clip" in self.preprocessors:
                #print("Clipping...")
                X[i,] = clipping(X[i,], sigma=4)
            if "rereference" in self.preprocessors:
                #print("Referencing...")
                X[i,] = rereference(X[i,])
            
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    def datagenerator(X_in, y_in, fs=250, n_crops=1, crop_len=3.5, batch_size=32, n_classes=4):
        while True:
            start = 0
            end = batch_size

            while start  < len(X_in): 
                # load your images from numpy arrays or read from directory
                X_out, y_out, _ = cropping(X_in[start:end, ...], y_in[start:end, ...], fs, n_crops, crop_len, shuffled=True)
                n_trials, n_channels, n_samples = X_out.shape
                X_out = np.reshape(X_out, (n_trials, 1, n_channels, n_samples)) 
            
                yield X_out, to_categorical(y_out, n_classes)

                start += batch_size
                end += batch_size
