import logging
import numpy as np
import pickle
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from .csp import CSP
from .fbcsp import FBCSP
from .riemann import Riemann
from .convnets import ShallowConvNet

# Reproducibility
seed_value = 0
np.random.seed(seed_value)


def get_CSP_model():
    model_name = 'CSP'
    search_space = {'classifier__C': (1e-3, 1e3, 'log-uniform')}
    model = Pipeline(steps=[('feat', CSP()),
                            ('classifier', SVC())])
    # ('classifier', SVC(kernel='rbf', gamma='scale', C=10))])
    return model, search_space, model_name


def get_FBCSP_model():
    model_name = 'FBCSP'
    search_space = {'classifier__C': (1e-1, 1e3, 'log-uniform')}
    model = Pipeline([('feat', FBCSP(fs=250, f_type='butter', m=2, k=-1)),
                      ('classifier', SVC(kernel='rbf', gamma='scale', C=10))])
    return model, search_space, model_name


def get_Riemann_model():
    model_name = 'Riemann'
    search_space = {'feat__f_order': (1, 5),
                    'feat__f_type': ['butter', 'cheby', 'ellip'],
                    'classifier__kernel': ['rbf', 'linear', 'poly'],
                    'classifier__degree': (1, 5),
                    'classifier__C': (1e-1, 1e3, 'log-uniform')}
    model = Pipeline([('feat', Riemann(fs=250)),
                      ('classifier', SVC(kernel='linear', gamma='scale', C=10))])
    return model, search_space, model_name


def get_ShallowConv_model():
    model_name = 'ShallowConv'
    search_space = {}
    model = ShallowConvNet()
    return model, search_space, model_name


def get_model(model_str):
    if model_str == 'CSP':
        model, search_space, model_name = get_CSP_model()
    elif model_str == 'FBCSP':
        model, search_space, model_name = get_FBCSP_model()
    elif model_str == 'Riemann':
        model, search_space, model_name = get_Riemann_model()
    elif model_str == 'ShallowConv':
        model, search_space, model_name = get_ShallowConv_model()
    else:
        raise RuntimeError('Invalid model selection')
    return model, search_space, model_name


def train(model_name, X_train, y_train, n_iters=1):
    # Extract MI phase
    fs = 250
    start = 2.5
    end = 6.
    logging.info(f'Extracting MI: [{start} to {end}]s')
    X_train = X_train[:, :, int(start*fs): int(end*fs)]

    # Load model
    model, search_space, model_name = get_model(model_name)
    start_time = time.time()

    # Define split
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf = list(skf.split(X_train, y_train))

    # Optimize
    logging.info('Starting Bayes optimization')
    gridSearch = BayesSearchCV(model, search_space, cv=skf, n_jobs=1,
                               refit=True, scoring='accuracy', n_iter=n_iters,)
    gridSearch.fit(X_train, y_train)

    scores = [gridSearch.cv_results_[f'split{k}_test_score'][gridSearch.best_index_]
              for k in range(gridSearch.n_splits_)]
    training_time = time.time() - start_time

    return gridSearch, np.mean(scores), np.std(scores), training_time


def save_model(model, save_path, pkl_filename):
    model_pkl = open(f'{save_path}/{pkl_filename}', 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def load_model(model_path):
    unpickle = open(model_path, 'rb')
    return pickle.load(unpickle)
