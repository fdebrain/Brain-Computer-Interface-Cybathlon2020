import logging
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import BayesSearchCV


def train(X_train, y_train, pipeline, search_space, mode, n_iters=10):
    logging.info(f'Training pipeline in {mode} mode '
                 f'using {len(X_train)} trials')
    start_time = time.time()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf = list(skf.split(X_train, y_train))

    if mode == 'optimize':
        pipeline = BayesSearchCV(pipeline, search_space, cv=skf, n_jobs=-1,
                                 refit=True, scoring='accuracy', n_iter=n_iters,
                                 verbose=True, random_state=0)
        pipeline.fit(X_train, y_train)

        # Extract cv validation scores
        logging.info(pipeline.cv_results_)
        cv_mean = pipeline.cv_results_[
            f'mean_test_score'][pipeline.best_index_]
        cv_std = pipeline.cv_results_[f'std_test_score'][pipeline.best_index_]
    elif mode == 'validate':
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=skf, n_jobs=-1,
                                 scoring='accuracy')
        logging.info(scores)
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time
    return pipeline, cv_mean, cv_std, training_time
