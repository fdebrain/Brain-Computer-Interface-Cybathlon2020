import logging
import numpy as np
import pickle
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from .csp import CSP
from .fbcsp import FBCSP
from .riemann import Riemann

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
    logging.info(f'Training {model_name} model in {mode} mode')
    start_time = time.time()
    model, search_space, model_name = get_model(model_name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    skf = list(skf.split(X_train, y_train))

    if mode == 'optimize':
        model = BayesSearchCV(model, search_space, cv=skf, n_jobs=1,
                              refit=True, scoring='accuracy', n_iter=n_iters,
                              verbose=True, random_state=0)
        model.fit(X_train, y_train)

        # Extract cv validation scores
        logging.info(model.cv_results_)
        cv_mean = model.cv_results_[f'mean_test_score'][model.best_index_]
        cv_std = model.cv_results_[f'std_test_score'][model.best_index_]
    elif mode == 'validate':
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, n_jobs=1,
                                 scoring='accuracy')
        logging.info(scores)
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    return model, cv_mean, cv_std, training_time


def save_model(model, save_path, pkl_filename):
    model_pkl = open(f'{save_path}/{pkl_filename}', 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def load_model(model_path):
    unpickle = open(model_path, 'rb')
    return pickle.load(unpickle)
