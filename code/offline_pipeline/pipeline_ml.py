import comet_ml
from comet_ml import Experiment, Optimizer

import matplotlib.pyplot as plt
import numpy as np
import os, random, time
import sys
sys.path.append('..')

from data_loading_functions.data_loader import EEGDataset, cropping
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from preprocessing_functions.preproc_functions import filtering
from feature_extraction_functions.csp import CSP
from feature_extraction_functions.fbcsp import FBCSP
from feature_extraction_functions.riemann import Riemann
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, cohen_kappa_score

import xgboost

from visualization_functions.metric_visualizers import plot_full_barchart, plot_cm
from visualization_functions.metric_visualizers import compute_online_metrics, plot_online_metrics
from visualization_functions.signal_visualizers import tsne_plot

# Reproducibility
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


class MLExperiment:
    def __init__(self, params):
        self.session_idx = params['session_idx'] 
        self.model_name = params['model_name']
        self.MI_labels = params['MI_labels']
        self.fs = params['fs']
        self.start = params['start']
        self.end = params['end']
        self.n_crops = params['n_crops']
        self.crop_len = params['crop_len']
        self.average_pred = params['average_pred']
        self.get_online_metrics = params['get_online_metrics']
        self.m = params['m']
        self.C = params['C']
        self.online_stride = self.fs//10
        self.n_folds = 5
        self.params = params # For logging purpose
    
        # Scores
        self.train_score = { 'accuracy' : [], 'kappa' : [] }
        self.test_score = { 'accuracy' : [], 'kappa' : [] }
        self.online_scores_train = { 'accuracy' : [], 'kappa' : [] }
        self.online_scores_test = { 'accuracy' : [], 'kappa' : [] }
        self.conf_matrices = {}
        self.train_times = []
        self.test_times = []
        self.inference_delays = []
        self.best_acc = -1

        # Load dataset (full trial and extracted MI ROI)
        self.dataloader_params = \
        {
            'data_path' : params['path'],
            'fs' : self.fs,
            'filt' : params['filt'],
            'rereferencing' : params['rereferencing'],
            'standardization' : params['standardization'],
            'valid_ratio' : 0.0,
            'load_test' : False,
            'balanced' : True
        }
        
    def run(self):
        print('Loading ', self.dataloader_params['data_path'])
        X_full, y, _, _, _, _ = EEGDataset(1, **self.dataloader_params).load_dataset()
        n_classes = len(np.unique(y))
        self.n_samples_tot = X_full.shape[-1]
        
        # Pre-processing - filtering
        if self.model_name=='CSP':
            X_full = filtering(X_full, self.fs, f_order=5, f_low=7, f_high=35, f_type='cheby')
        
        # Extract motor imagery phase
        X = X_full[:, :, int(self.start*self.fs): int(self.end*self.fs)]

        ## Reject trials
        #def detect_peak(X, threshold=150):
        #    val_min = np.min(X)
        #    val_max = np.max(X)
        #    return np.abs(val_min) + np.abs(val_max) > threshold
    
        #mask = [ False if detect_peak(x) else True for x in X ]
        #print('Rejected {} trials'.format(len(X) - sum(mask)))
        #X, y = X[mask], y[mask]

        # For logging purpose (t-SNE)
        self.X = X
        self.y = y

        # Get more robust validation metrics via k-folding (no pre-defined test set like in BCIC)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=1)
        idx_kfold = 0
        for idx_train, idx_test in skf.split(X, y):
            # Get k-fold data (with cropping)
            print('Fold ', idx_kfold+1)
            idx_kfold += 1
            X_train, y_train = X[idx_train], y[idx_train]
            X_test, y_test = X[idx_test], y[idx_test]
            X_online_train, y_online_train = X_full[idx_train], y[idx_train]
            X_online_test, y_online_test = X_full[idx_test], y[idx_test]

            # Cropping the data
            X_train_crops, y_train_crops, trial2crops_train = cropping(X_train, y_train, self.fs, self.n_crops,
                                                                       self.crop_len, shuffled=True)
            X_test_crops, y_test_crops, trial2crops_test = cropping(X_test, y_test, self.fs, self.n_crops,
                                                                    self.crop_len, shuffled=False)

            # Define pipeline (feature extraction and classifier)
            assert self.model_name in ['CSP', 'FBCSP', 'Riemann','XGB'], "Choose model among {'CSP', 'FBCSP', 'Riemann', 'XGB'}"
            if self.model_name=='CSP':
                feature_extractor = CSP(n_classes, self.m, regularize_cov=False)
                classifier = SVC(self.C, 'rbf', gamma='scale', random_state=1)
            elif self.model_name=='FBCSP':
                feature_extractor = FBCSP(self.fs, n_classes, self.m, k=-1)
                classifier = SVC(self.C, 'rbf', gamma='scale', random_state=1)
            elif self.model_name=='Riemann':
                feature_extractor = Riemann(self.fs, n_classes)
                classifier = LinearSVC(C=self.C, random_state=1)
            elif self.model_name=='XGB':
                feature_extractor = CSP(n_classes, self.m, regularize_cov=False)
                classifier = xgboost.XGBClassifier(objective="multi:softprob", random_state=1, max_depth=7, n_estimators=500, seed=1)
            self.model = Pipeline(steps=[('feat', feature_extractor), ('clf', classifier)])

            # Training
            train_time = time.time()
            self.model.fit(X=X_train_crops, y=y_train_crops)
            train_time = time.time() - train_time
            self.train_times.append(1e3*train_time/X_train.shape[0])
            print('Took {:.1f}ms to fit the model (about {:.1f}ms per trial).'.format(1e3*train_time, 
                                                                                  1e3*train_time/X_train.shape[0]))

            # Validation (train set metrics) 
            y_pred_crops = self.model.predict(X_train_crops)
            if self.average_pred:
                y_pred_train = np.empty_like(y_train)
                for trial_idx in range(len(y_train)):
                    labels, cnts = np.unique(y_pred_crops[trial2crops_train[trial_idx]], return_counts=True)
                    y_pred_train[trial_idx] = labels[np.argmax(cnts)]
                y_true_train = y_train
            else:
                y_pred_train = y_pred_crops
                y_true_train = y_train_crops

            # Validation (test set metrics with prediction averaging)
            test_time = time.time()
            y_pred_crops = self.model.predict(X_test_crops)
            if self.average_pred:
                y_pred_test = np.empty_like(y_test)
                for trial_idx in range(len(y_test)):
                    labels, cnts = np.unique(y_pred_crops[trial2crops_test[trial_idx]], return_counts=True)
                    y_pred_test[trial_idx] = labels[np.argmax(cnts)]
                y_true_test = y_test
            else:
                y_pred_test = y_pred_crops
                y_true_test = y_test_crops
            test_time = time.time() - test_time
            self.test_times.append(1e3*test_time/X_test.shape[0])
            print('Took {:.1f}ms to perform inference (about {:.1f}ms per trial).'.format(1e3*test_time, 
                                                                                      1e3*test_time/X_test.shape[0]))

            # Saving metrics and confusion matrices for each k-fold
            self.train_score['accuracy'].append(accuracy_score(y_true_train, y_pred_train))
            self.train_score['kappa'].append(cohen_kappa_score(y_true_train, y_pred_train))
            self.test_score['accuracy'].append(accuracy_score(y_true_test, y_pred_test))
            self.test_score['kappa'].append(cohen_kappa_score(y_true_test, y_pred_test))
            self.conf_matrices['y_pred{}'.format(idx_kfold)] = y_pred_test if self.average_pred else y_pred_crops 
            self.conf_matrices['y_true{}'.format(idx_kfold)] = y_true_test

            # Save current test set for visualization purpose (t-SNE)
            if self.test_score['accuracy'][-1] > self.best_acc:
                self.best_acc = self.test_score['accuracy'][-1]
                self.X_best = X_test_crops
                self.y_best = y_test_crops
            
            # Getting online scores (temporal accuracy averaged over trials)
            if self.get_online_metrics:
                scores_train, _ = compute_online_metrics(self.model, X_online_train, y_online_train, 
                                                         int(self.crop_len*self.fs), self.online_stride)
                if np.size(self.online_scores_train['accuracy'])==0:
                    self.online_scores_train = scores_train
                else:
                    self.online_scores_train['accuracy'] += scores_train['accuracy']
                    self.online_scores_train['kappa'] += scores_train['kappa']

                
                scores_test, tmp = compute_online_metrics(self.model, X_online_test, y_online_test, 
                                                          int(self.crop_len*self.fs), self.online_stride)
                self.inference_delays.append(1e3*tmp)

                if np.size(self.online_scores_test['accuracy'])==0:
                    self.online_scores_test = scores_test
                else:
                    self.online_scores_test['accuracy'] += scores_test['accuracy']
                    self.online_scores_test['kappa'] += scores_test['kappa']

            # Logging for current Kfold
            print('Pilot 1 - Train: {:.3f}/{:.3f} - Test: {:.3f}/{:.3f}. \n'.format(self.train_score['accuracy'][-1],
                                                                                    self.train_score['kappa'][-1],
                                                                                    self.test_score['accuracy'][-1],
                                                                                    self.test_score['kappa'][-1]))

        print('Mean training score: {:.3f} (accuracy) - {:.3f} (kappa)'.format(np.mean(self.train_score['accuracy']),
                                                                              np.mean(self.train_score['kappa'])))
        print('Mean test score: {:.3f}+-{:.3f} (accuracy) - {:.3f}+-{:.3f} (kappa)'.format(
                                                                              np.mean(self.test_score['accuracy']),
                                                                              np.std(self.test_score['accuracy']),
                                                                              np.mean(self.test_score['kappa']),
                                                                              np.std(self.test_score['kappa'])))
        print('Mean training time per trial: {:.1f}ms'.format(np.mean(self.train_times)))
        print('Mean testing time per trial: {:.1f}ms'.format(np.mean(self.test_times)))
        print('Mean inference delay: {:.1f}ms'.format(np.mean(self.inference_delays))) if self.get_online_metrics \
                                                                                       else ''
        
    def plot_results(self, get_tsne=False, full_data=False):
        # Barchart
        fig, ax = plt.subplots()
        fig = plot_full_barchart(self.test_score, n_pilots=self.n_folds, 
                                 title='{} model - Competition Dataset - Session {}'.format(self.model_name, self.session_idx),
                                 fig=fig)
        fig.get_axes()[0].set_xlabel('Fold')

        # Confusion matrices
        fig, ax = plt.subplots(2, self.n_folds//2+1, figsize=(9,5))
        [ plot_cm(self.conf_matrices, idx_fold, fig=fig, ax_idx=idx_fold-1, title='Fold {}'.format(idx_fold),
                  class_names=self.MI_labels) for idx_fold in range(1, self.n_folds+1)]
        fig.delaxes(ax[1,2])
        fig.tight_layout()

        # Online metrics over full trial
        if self.get_online_metrics:
            fig, ax = plt.subplots(1,2, figsize=(8,4))
            plot_online_metrics(self.online_scores_train, len_tot=self.n_samples_tot,
                                n_folds=self.n_folds, stride=self.online_stride, 
                                title='{} model - Competition Dataset \n Session {} (Train)'.format(self.model_name, self.session_idx), 
                                fig=fig, ax_idx=0);

            plot_online_metrics(self.online_scores_test, len_tot=self.n_samples_tot,
                                n_folds=self.n_folds, stride=self.online_stride, 
                                title='{} model - Competition Dataset \n Session {} (Test)'.format(self.model_name, self.session_idx),
                                fig=fig, ax_idx=1);
            fig.tight_layout()

        if get_tsne:
            X = self.X if full_data else self.X_best
            y = self.y if full_data else self.y_best
            
            fig, ax = plt.subplots(1, 2, figsize=(9,5))
            tsne_plot(X, y, title="t-SNE - Pilot 1 - Raw", label_names=self.MI_labels, fig=fig, ax_idx=0)
            tsne_plot(self.model.named_steps['feat'].transform(X), y, label_names=self.MI_labels,
                      title="t-SNE - Pilot 1 - After {}".format(self.model_name), fig=fig, ax_idx=1)
            fig.tight_layout()

    def logging(self):
        experiment = Experiment(api_key='cSZq9kuH2I87ezvm2dEWTx6op', project_name='Session {} - {}'.format(self.session_idx, self.model_name),
                                auto_metric_logging=False, auto_param_logging=False, log_code=False)

        # Parameters
        experiment.log_parameters(self.params)
        
        # Scalar metrics
        experiment.log_metrics({'Average accuracy' : np.mean(self.test_score['accuracy']), 
                                'Std accuracy' : np.std(self.test_score['accuracy']),
                                'Average train time per trial (ms)' : np.mean(self.train_times), 
                                'Average test time per trial (ms)' : np.mean(self.test_times)})
        
        # Barchart
        fig = plot_full_barchart(self.test_score, n_pilots=self.n_folds, title='{} - Session {}'.format(self.model_name, self.session_idx))
        fig.get_axes()[0].set_xlabel('Fold')
        experiment.log_figure(figure_name='Accuracy barchart', figure=fig)

        # Confusion matrices
        fig, ax = plt.subplots(2, self.n_folds//2+1, figsize=(9,5))
        [ plot_cm(self.conf_matrices, idx_fold, fig=fig, ax_idx=idx_fold-1, title='Fold {}'.format(idx_fold), class_names=self.MI_labels) 
        for idx_fold in range(1, self.n_folds+1)]
        fig.delaxes(ax[1,2])
        fig.tight_layout()
        experiment.log_figure(figure_name='Confusion matrices', figure=fig)

        # Online metrics
        if self.get_online_metrics:
            experiment.log_metrics({'Average inference delay' : np.mean(self.inference_delays), 
                                    'Std inference delay' : np.std(self.inference_delays)})

            fig, ax = plt.subplots(1,2, figsize=(8,4))
            plot_online_metrics(self.online_scores_train, len_tot=self.n_samples_tot, n_folds=self.n_folds, stride=self.online_stride, 
                                title='{} model - Competition Dataset \n Session {} (Train)'.format(self.model_name, self.session_idx),
                                fig=fig, ax_idx=0);
            plot_online_metrics(self.online_scores_test, len_tot=self.n_samples_tot, n_folds=self.n_folds, stride=self.online_stride, 
                                title='{} model - Competition Dataset \n Session {} (Test)'.format(self.model_name, self.session_idx),
                                fig=fig, ax_idx=1);
            fig.tight_layout()
            experiment.log_figure(figure_name='Online evaluation', figure=fig)

        experiment.end()
