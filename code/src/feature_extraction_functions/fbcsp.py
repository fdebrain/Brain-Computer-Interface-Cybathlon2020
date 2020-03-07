import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from ..preprocessing import filtering
from .csp import CSP


class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, fs=250, n_classes=4, m=2, f_order=2, f_type='butter', k=-1, freq_bands=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]):
        self.fs = fs
        self.m = m
        self.n_classes = n_classes
        self.freq_bands = freq_bands
        self.f_order = f_order
        self.f_type = f_type
        self.k = k

    def fit(self, X, y):
        '''
        Apply filter bank to input EEG signal, fit each CSP block and the MIBIF feature selection.
        Input:
            X: EEG data in numpy format (trials, channels, samples).
            y: EEG labels numpy format (trial).
        '''

        self.csp_blocks = []
        feats = []
        for f_band in self.freq_bands:
            # Filter signal on given frequency band
            X_filt = filtering(X, fs=self.fs, f_order=self.f_order,
                               f_low=f_band[0], f_high=f_band[1],
                               f_type=self.f_type)

            # Apply CSP and save block (with trained filters)
            csp = CSP(n_classes=self.n_classes, m=self.m)
            csp_feats = csp.fit_transform(X_filt, y)  # (n_trials, n_csp)

            feats = csp_feats if len(
                feats) == 0 else np.hstack([feats, csp_feats])
            self.csp_blocks.append(csp)

        self.n_csp = self.csp_blocks[-1].n_csp
        self.n_feats_tot = len(self.freq_bands) * self.n_csp
        assert feats.shape == (X.shape[0], self.n_feats_tot)

        # Feature selection based on MIBIF algorithm
        if self.k > 0:
            self.feature_selection = SelectKBest(
                mutual_info_classif, k=self.k, )
            self.feature_selection.fit(feats, y)

        return self

    def transform(self, X):
        '''
        Apply filter bank and CSP transform to input EEG signal and selection features.
        Input:
            - X: array of shape (n_trials, n_channels, n_samples).
        Output:
            - selected_feats: extracted features of shape (n_trials, 2*k).
        '''
        feats = []
        m = self.m
        for band_idx, f_band in enumerate(self.freq_bands):
            # Filter input signal on given frequency band
            X_filt = filtering(X, fs=self.fs, f_order=self.f_order,
                               f_low=f_band[0], f_high=f_band[1],
                               f_type=self.f_type)

            # Compute CSP features associated to current frequency band
            csp_feats = self.csp_blocks[band_idx].transform(X_filt)

            # Concatenate CSP features
            feats = csp_feats if len(
                feats) == 0 else np.hstack([feats, csp_feats])

        # Select k best features and their pairs
        selected_feats = []
        if self.k > 0:
            select_idxs = self.feature_selection.get_support(indices=True)
            for idx in select_idxs:
                # feature index inside 2*m block of features
                sub_idx = idx % (2*m)
                # corresponding paired index
                comp_idx = idx + (m if sub_idx < m else -m)

                pair_feats = np.hstack([feats[:, idx].reshape(-1, 1),
                                        feats[:, comp_idx].reshape(-1, 1)])
                selected_feats = pair_feats if len(selected_feats) == 0 \
                    else np.hstack([selected_feats, pair_feats])
        else:
            selected_feats = feats
        return selected_feats
