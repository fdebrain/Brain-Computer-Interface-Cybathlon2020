import logging
import os
import pickle
import time
import json
from collections import Counter

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import BayesSearchCV

from .feature_extraction_functions.csp import CSP
from .feature_extraction_functions.fbcsp import FBCSP
from .feature_extraction_functions.riemann import Riemann
from .feature_extraction_functions.convnets import ShallowConvNet

# Reproducibility
seed_value = 0
np.random.seed(seed_value)


def get_CSP_model():
    model_name = 'CSP'
    search_space = {'classifier__C': (1e-3, 1e3, 'log-uniform')}
    model = Pipeline(steps=[('feat', CSP()),
                            ('classifier', SVC())])
    return model, search_space, model_name


def get_FBCSP_model():
    model_name = 'FBCSP'
    search_space = {'classifier__C': (1e-2, 1e3, 'log-uniform')}
    model = Pipeline([('feat', FBCSP(fs=500, f_type='butter', m=2, k=-1)),
                      ('classifier', SVC(kernel='rbf', gamma='scale', C=10))])
    return model, search_space, model_name


def get_Riemann_model():
    model_name = 'Riemann'
    search_space = {'feat__f_order': (1, 5),
                    'feat__f_type': ['butter', 'cheby', 'ellip'],
                    'classifier__kernel': ['rbf', 'linear', 'poly'],
                    'classifier__degree': (1, 5),
                    'classifier__C': (1e-1, 1e3, 'log-uniform')}
    model = Pipeline([('feat', Riemann(fs=500)),
                      ('classifier', SVC(kernel='linear', gamma='scale', C=10))])
    return model, search_space, model_name


def get_model(model_str, **params):
    if model_str == 'CSP':
        model, search_space, model_name = get_CSP_model()
    elif model_str == 'FBCSP':
        model, search_space, model_name = get_FBCSP_model()
    elif model_str == 'Riemann':
        model, search_space, model_name = get_Riemann_model()
    else:
        raise RuntimeError('Invalid model selection')
    return model, search_space, model_name


def train(model_name, X_train, y_train, mode, n_iters=10):
    logging.info(f'Training {model_name} model in {mode} mode '
                 f'using {len(X_train)} trials')
    start_time = time.time()
    model, search_space, model_name = get_model(model_name)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf = list(skf.split(X_train, y_train))

    if mode == 'optimize':
        model = BayesSearchCV(model, search_space, cv=skf, n_jobs=-1,
                              refit=True, scoring='accuracy', n_iter=n_iters,
                              verbose=True, random_state=0)
        model.fit(X_train, y_train)

        # Extract cv validation scores
        logging.info(model.cv_results_)
        cv_mean = model.cv_results_[f'mean_test_score'][model.best_index_]
        cv_std = model.cv_results_[f'std_test_score'][model.best_index_]
    elif mode == 'validate':
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, n_jobs=-1,
                                 scoring='accuracy')
        logging.info(scores)
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    return model, cv_mean, cv_std, training_time


def predict(X, model, is_convnet):
    """Return prediction of a trained model given input EEG data.

    Arguments:
        X {np.array} -- EEG array of shape (n_trials, n_channels, n_samples)
        model {object} -- Trained model
        is_convnet {bool} -- Model is convNet

    Returns:
        np.array -- Array of predictions
    """
    y_preds = model.predict(X)

    # ConvNet case - Convert probabilities to int
    if is_convnet:
        logging.info(y_preds)
        y_preds = np.argmax(y_preds, axis=1)

    y_pred = Counter(y_preds).most_common()[0][0]

    # TODO: If less than n_thresh occurences, return 'Rest' action
    return y_pred


def load_model(model_path):
    """Load pre-trained model {FBCSP, Riemann, ShallowConvNet)

    Arguments:
        model_path {Path} -- Model path to load

    Raises:
        ValueError: Model suffix not recognised

    Returns:
        object -- Model
    """
    logging.info(f'Loading model {model_path}')

    if model_path.suffix == '.pkl':
        unpickle = open(model_path, 'rb')
        model = pickle.load(unpickle)
    elif model_path.suffix == '.h5':
        model = ShallowConvNet(n_channels=61,
                               n_samples=250)
        model.load_weights(model_path)
    else:
        raise ValueError('Model format not recognized !')

    logging.info(f'Successfully loaded model: {model}')
    return model


def save_model(model, save_path, filename):
    """Save model as .pkl file.

    Arguments:
        model {object} -- Fitted model
        save_path {Path} -- Path of save directory
        filename {str} -- Name of file to save
    """
    if not os.path.isdir(save_path):
        logging.info(f'Creating directory {save_path}')
        os.mkdir(save_path)

    logging.info(f'Saving model {filename}.pkl')
    with open(save_path / f'{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info('Successfully saved model !')


def save_json(model_info, save_path, filename):
    with open(save_path / f'{filename}.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)
