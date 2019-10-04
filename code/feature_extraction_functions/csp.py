import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_classes=4, m=2, regularize_cov=False):
        self.m = m
        self.n_classes = n_classes
        self.regularize_cov = regularize_cov
        self.rho = 0.1
        
    def fit(self, X, y):
        ''' 
        Compute CSP filters that optimize the variance ratios between trials of different classes.
        Input:
            X: Filtered EEG data with zero-mean component in numpy format (trials, channels, samples).
            y: EEG labels numpy format (trial).
        '''
        n_classes = self.n_classes
        m = self.m
        n_trials, n_channels, n_samples = X.shape
        labels_list = np.unique(y)
        assert len(set(y)) == n_classes
        
        # Compute covariance matrix for each trial
        if self.regularize_cov:
            cov_matrices = np.array([(1/(n_samples-1))*np.dot(X_t, X_t.T) + (self.rho/n_samples)*np.eye(n_channels) for X_t in X])
        else:
            cov_matrices = np.array([X_t @ X_t.T for X_t in X])
            cov_matrices = np.array([C/np.trace(C) for C in cov_matrices])

        # Averaging covariance matrices of the same class 
        cov_avg = np.zeros((n_classes, n_channels, n_channels), dtype=np.float)
        for c in range(n_classes):
            idxs = np.where(y==labels_list[c])[0]
            cov_avg[c] = np.sum(cov_matrices[idxs], axis=0) / len(idxs)

        # Generalized Eigenvalue Decomposition (compare classes 2 by 2) 
        self.w = []
        self.pairs_idx = []
        for c1 in range(n_classes):
            for c2 in range(c1 + 1, n_classes):
                # Solve C_1*U = C_2*U*D where D = diag(w) and U = vr
                eig_vals, U = scipy.linalg.eig(cov_avg[c1], cov_avg[c2])
                assert np.allclose(cov_avg[c1] @ U - cov_avg[c2] @ U @ np.diag(eig_vals), np.zeros((n_channels, n_channels)))
                
                # Sort eigenvalues and pair i-th biggest with i-th smallest 
                eig_vals_abs = np.abs(eig_vals)
                sorted_idxs = np.argsort(eig_vals_abs) # (increasing order)
                
                # Extract corresponding eigenvectors (spatial filters)
                chosen_idxs = np.zeros(2*m, dtype=np.int16)
                chosen_idxs[:m] = sorted_idxs[:m] # m smallest
                chosen_idxs[m:2*m] = sorted_idxs[-m:] # m biggest
                eig_vecs = U[:, chosen_idxs]
                
                # Stack these 2*m spatial filters horizontally with the previous ones
                self.w = eig_vecs if len(self.w)==0 else np.hstack([self.w, eig_vecs])
                self.pairs_idx.append(chosen_idxs) # elements {i, i+m} go in pairs
                
        self.n_csp = 2*m*np.sum(range(n_classes))
        assert self.w.shape == (n_channels, self.n_csp), 'Got w of shape {} instead of {}.'.format(self.w.shape, [n_channels, self.n_csp])
        self.w = self.w.astype(float)

        return self
             
    def transform(self, X):
        ''' 
        Apply CSP transform on the input EEG data and compute the log-variance features.
        Input: 
            - X: EEG array of shape (n_trials, n_channels, n_samples).
        Output:
            - feats: extracted features of shape (n_trials, n_csp).
        '''
        n_trials, n_channels, n_samples = X.shape
        n_csp = self.n_csp
        
        # Apply spatial transformation to input signal using the previously computed CSP filters
        X_transformed = np.array([self.w.T @ X[trial_idx] for trial_idx in range(X.shape[0])])
        assert X_transformed.shape == (n_trials, n_csp, n_samples)
                
        # Compute variance of each row of the CSP-transformed signal (apply on temporal axis)
        variances = np.array([ X_transformed[trial_idx] @ X_transformed[trial_idx].T  for trial_idx in range(n_trials) ])
        assert variances.shape == (n_trials, n_csp, n_csp)
    
        # Compute normalized log-variance features 
        feats = np.array([ np.log10(np.diag(variances[trial_idx]) / np.trace(variances[trial_idx])) for trial_idx in range(n_trials) ])
        assert feats.shape == (n_trials, n_csp)

        return feats 
