import numpy as np
import time
import scipy.signal
from preprocessing_functions.preproc_functions import filtering

# Source: https://github.com/alexandrebarachant/pyRiemann/blob/master/pyriemann/utils/base.py
def _matrix_operator(C, operator):
    """ Matrix equivalent of an operator. """
    if not np.isfinite(C).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")
    eigvals, eigvects = scipy.linalg.eigh(C, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    C_out = eigvects @ eigvals @ eigvects.T
    return C_out

def logm(C):
    """ Return the matrix logarithm of a covariance matrix. """
    return _matrix_operator(C, np.log)

def expm(C):
    """ Return the matrix exponential of a covariance matrix. """
    return _matrix_operator(C, np.exp)

def sqrtm(C):
    """ Return the matrix square root of a covariance matrix. """
    return _matrix_operator(C, np.sqrt)

def invsqrtm(C):
    """ Return the inverse matrix square root of a covariance matrix """
    isqrt = lambda x: 1. / np.sqrt(x)
    return _matrix_operator(C, isqrt)

def half_vectorization(C):
        '''
        Calculates half vectorization of a matrix.
        Input:
            - C: SPD matrix of shape (n_channel,n_channel)
        Output: 
            - C_vec: Vectorized matrix of shape n_riemann
        '''
        n_channels, _ = C.shape 
        n_elements = int( (n_channels + 1) * n_channels / 2 )
        C_vec = np.zeros(n_elements)
        
        # Diagonal elements 
        C_vec[:n_channels] = np.diag(C)
        
        # Off-diagonal elements in upper matrix (multiply by sqrt(2) to conserve norm)
        sqrt2 = np.sqrt(2)  
        tmp = np.triu(C, k=1).flatten()
        C_vec[n_channels:] = sqrt2 * tmp[tmp!=0]
        #assert np.isclose(np.linalg.norm(C, ord='fro') - np.linalg.norm(C_vec), 0)
        
        return C_vec    


class Riemann:
    def __init__(self, fs=250, n_classes=4, f_order=2, f_type='butter', f_min=4, f_max=40, bandwidths=[2,4,8,16,32]):
        self.fs = fs
        self.n_classes = n_classes
        self.f_order = f_order
        self.f_type = f_type
        self.rho = 0.1
        self.f_bands = self.load_bands(bandwidths, f_min, f_max, f_order, f_type)
        
    def load_bands(self, bandwidths, f_min, f_max, f_order, f_type):
        ''' Initialize filter bank bands.
        Inputs:
            - bandwidths: List of filter bandwidths (array of int).
            - f_min, f_max: minimal and maximal filter frequencies (int).
            - f_order: filter order (int).
            - f_type: filter type {'butter', 'cheby', 'ellip'} (string).
        Output:
            - f_bands: filter bank bands (array of shape (n_bands, 2)).
        '''
        
        f_bands = []
        for bw in bandwidths:
            f = f_min
            while f + bw <= f_max:
                f_bands.append([ f, f + bw ])
                f += 2 if bw<4 else 4
        f_bands = np.array(f_bands)
        return f_bands

    def fit(self, X, y):
        ''' 
        Apply filtering to input signal and compute regularized covariance matrices. 
        Compute the reference matrices of each filter block. 
        Input:
            X: EEG data in numpy format (trials, channels, samples).
            y: EEG labels numpy format (trial).
        '''
        now = time.time()
        n_trials, n_channels, n_samples = X.shape
        self.C_ref_invsqrt = np.zeros((len(self.f_bands), n_channels, n_channels))

        for band_idx, f_band in enumerate(self.f_bands):
            # Band-pass filtering input signal
            X_filt = filtering(X, fs=self.fs, f_order=self.f_order, f_low=f_band[0], f_high=f_band[1], f_type=self.f_type)
                       
            # Compute covariance matrices  (regularized version)
            cov_matrices = np.array([(1/(n_samples-1))*np.dot(X_t, X_t.T) + (self.rho/n_samples)*np.eye(n_channels) for X_t in X_filt])

            # Compute Riemannian mean covariance matrix inside current filter block.
            C_ref = np.mean(cov_matrices, axis=0)

	    # Compute C_ref^(-1/2) of current filter block
            self.C_ref_invsqrt[band_idx] = invsqrtm(C_ref)
            
        #print('Took {:.1f}s to fit model'.format(time.time() - now))
        return self
        
    def transform(self, X):
        """ 
        Compute multiscale riemannian features, i.e. the vectorized covariance matrices of each filter block projected in the Riemannian tangent space.  
        Input: 
            - X: EEG array of shape (n_trials, n_channels, n_samples).
        Output:
            - feats: extracted features of shape (n_trials, n_features).
        """
        n_trials, n_channels, n_samples = X.shape
        
        feats = []
        now = time.time()
        for band_idx, f_band in enumerate(self.f_bands):
            # Band-pass filtering input signal
            X_filt = filtering(X, fs=self.fs, f_order=self.f_order, f_low=f_band[0], f_high=f_band[1], f_type=self.f_type)
        
            # Compute covariance matrices (regularized version)
            cov_matrices = np.array([(1/(n_samples-1))*np.dot(X_t, X_t.T) + (self.rho / n_samples)*np.eye(n_channels) for X_t in X_filt])
            
            # Project in tangent space w.r.t C_ref
            c_ref_invsqrt = self.C_ref_invsqrt[band_idx]
            S_projections = np.array([ logm(c_ref_invsqrt @ cov_matrices[trial_idx] @ c_ref_invsqrt) for trial_idx in range(n_trials) ])
            
            # Vectorize projected matrices
            S_projections_vec = np.array([ half_vectorization(S_projections[trial_idx]) for trial_idx in range(n_trials) ])
            
            feats = S_projections_vec if len(feats)==0 else np.hstack([feats, S_projections_vec])

        #print('Took {:.1f}s to perform inference'.format(time.time() - now))        
        return feats
