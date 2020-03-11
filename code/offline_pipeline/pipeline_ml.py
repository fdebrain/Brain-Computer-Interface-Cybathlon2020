import os
import random
import logging
import traceback

import numpy as np

from src.dataloader import load_session, preprocessing
from src.preprocessing import cropping
from src.preprocessing import filtering
from src.models import train
from visualization_functions.metric_visualizers import (plot_cm,
                                                        plot_full_barchart,
                                                        plot_online_metrics)

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Reproducibility
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


class MLExperiment:
    def __init__(self, params):
        self.data_path = params["data_path"]
        self.model_name = params['model_name']
        self.MI_labels = params['MI_labels']
        self.train_mode = params['train_mode']
        self.n_iters = params['n_iters']

        # Preprocessing settings
        self.should_reref = params['rereferencing']
        self.should_standardize = params['standardization']
        self.should_filter = params['filt']
        self.start = params['start']
        self.end = params['end']
        self.n_crops = params['n_crops']
        self.crop_len = params['crop_len']

        # Scores
        self.train_score = {'accuracy': [], 'kappa': []}
        self.test_score = {'accuracy': [], 'kappa': []}
        self.online_scores_train = {'accuracy': [], 'kappa': []}
        self.online_scores_test = {'accuracy': [], 'kappa': []}
        self.conf_matrices = {}
        self.train_times = []
        self.test_times = []
        self.inference_delays = []
        self.best_acc = -1

    def run(self):
        X, y = {}, {}
        filepath = f'{self.data_path}/train/train1.npz'
        X, y, self.fs, self.ch_names = load_session(filepath,
                                                    start=self.start,
                                                    end=self.end)
        logging.info(f'Shape: X_full {X.shape} - y {y.shape}')

        # Pre-processing - filtering
        if self.model_name == 'CSP':
            X = filtering(X, self.fs, f_order=5,
                          f_low=7, f_high=35, f_type='cheby')

        # Cropping
        if self.n_crops > 1:
            X, y = cropping(X, y, self.fs,
                            n_crops=self.n_crops,
                            crop_len=self.crop_len)

        # Preprocessing
        X = preprocessing(X, self.fs,
                          rereference=self.should_reref,
                          filt=self.should_filter,
                          standardize=self.should_standardize)

        # For logging purpose (t-SNE)
        try:
            trained_model, cv_mean, cv_std, train_time = train(self.model_name,
                                                               X, y,
                                                               self.train_mode,
                                                               n_iters=self.n_iters)
        except Exception:
            logging.info(f'Training failed - {traceback.format_exc()}')

        logging.info(f'Trained successfully in {train_time:.0f}s \n'
                     f'Accuracy: {cv_mean:.2f}+-{cv_std:.2f} \n'
                     f'{trained_model}')


# if __name__ == '__main__':
#     params = {'data_path': '../Datasets/Pilots/Pilot_2/Session_15/formatted_filt_500Hz',
#               'model_name': 'FBCSP',
#               'MI_labels': ['Rest', 'Left', 'Right', 'Both'],
#               'train_mode': 'validate',
#               'n_iters': 5,
#               'rereferencing': False,
#               'standardization': False,
#               'filt': False,
#               'start': 2.0,
#               'end': 6.0,
#               'n_crops': 1,
#               'crop_len': 0.5}

#     exp = MLExperiment(params)
#     exp.run()
