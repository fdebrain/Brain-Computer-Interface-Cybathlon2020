
import logging
import sys
from pathlib import Path

import numpy as np
from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, Button, CheckboxButtonGroup, CheckboxGroup
from bokeh.models.widgets import Slider
from bokeh.layouts import row, column
from data_loading_functions.data_loader import EEGDataset

from src.models import train, save_model


class TrainerWidget:
    def __init__(self):
        self.data_path = Path('../Datasets/Pilots/Pilot_2')
        self.X, self.y = None, None
        self.fs = 500

    @property
    def available_sessions(self):
        sessions = self.data_path.glob('*')
        return [''] + [s.name for s in sessions]

    @property
    def train_ids(self):
        selected_ids = [self.select_session.labels[i]
                        for i in self.select_session.active]
        return selected_ids

    @property
    def selected_preproc(self):
        active = self.checkbox_preproc.active
        return [self.checkbox_preproc.labels[i] for i in active]

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def model_name(self):
        return self.select_model.value

    @property
    def train_mode(self):
        return 'optimize' if 'Optimize' in self.selected_settings \
            else 'validate'

    @property
    def start(self):
        return self.slider_roi_start.value

    @property
    def end(self):
        return self.slider_roi_end.value

    @property
    def n_iters(self):
        return self.slider_n_iters.value

    def on_model_change(self, attr, old, new):
        logging.info(f'Select model {new}')
        self.update_widget()

    def on_session_change(self, attr, old, new):
        logging.info(f"Select train sessions {new}")
        self.update_widget()

    def update_widget(self):
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'

    def on_train_start(self):
        assert self.model_name != '', 'Please select a model !'
        assert len(self.train_ids) > 0, 'Please select at least one session !'

        self.button_train.button_type = 'warning'
        self.button_train.label = 'Loading data...'
        curdoc().add_next_tick_callback(self.on_load)

    def on_load(self):

        # TODO: Write cleaner dataloader
        X, y = {}, {}
        for id in self.train_ids:
            logging.info(f'Loading {id}')
            self.dataloader_params = {
                'data_path': self.data_path / id / f'formatted_filt_{self.fs}Hz',
                'fs': self.fs,
                'filt': 'Filter' in self.selected_preproc,
                'rereferencing': 'Rereference' in self.selected_preproc,
                'standardization': 'Standardize' in self.selected_preproc,
                'valid_ratio': 0.0,
                'load_test': False,
                'balanced': True}
            dataset = EEGDataset(pilot_idx=1, **self.dataloader_params)

            try:
                X[id], y[id], _, _, _, _ = dataset.load_dataset()
                logging.info(f'{X[id].shape} - {y[id].shape}')
            except Exception as e:
                logging.info(f'Loading data failed - {e}')
                self.button_train.button_type = 'danger'
                self.button_train.label = 'Training failed'
                return

        # Concatenate all data
        self.X = np.vstack([X[id] for id in X.keys()])
        self.y = np.hstack([y[id] for id in y.keys()]).flatten()

        # Update session info
        self.div_info.text = f'<b>Sampling frequency:</b> {self.fs} Hz<br>' \
            f'<b>Classes:</b> {np.unique(self.y)} <br>' \
            f'<b>Nb trials:</b> {len(self.y)} <br>' \
            f'<b>Nb channels:</b> {self.X.shape[1]} <br>' \
            f'<b>Trial length:</b> {self.X.shape[-1] / self.fs}s <br>'

        self.button_train.label = 'Training...'
        curdoc().add_next_tick_callback(self.on_train)

    def on_train(self):
        logging.info(f'Extracting MI: [{self.start} to {self.end}]s')
        self.X = self.X[:, :, int(self.start*self.fs): int(self.end*self.fs)]

        # Instanciate and train model
        try:
            trained_model, cv_mean, cv_std, train_time = train(self.model_name,
                                                               self.X, self.y,
                                                               self.train_mode,
                                                               n_iters=1)
        except Exception as e:
            logging.info(f'Training failed - {e}')
            self.button_train.button_type = 'danger'
            self.button_train.label = 'Failed'
            return

        logging.info(f'Trained successfully in {train_time:.0f}s \n'
                     f'Accuracy: {cv_mean:.2f}+-{cv_std:.2f} \n'
                     f'{trained_model}')

        model_to_save = trained_model if self.train_mode == 'validate' \
            else trained_model.best_estimator_

        if 'Save' in self.selected_settings:
            save_path = './saved_models'
            win_len = int(self.fs*(self.end-self.start))
            dataset_name = '_'.join([id for id in self.train_ids])
            pkl_filename = f"{self.model_name}_{dataset_name}_{win_len}.pkl"
            save_model(model_to_save, save_path, pkl_filename)

        # Update info
        self.button_train.button_type = 'success'
        self.button_train.label = 'Trained'
        self.div_info.text += f'<b>Accuracy:</b> {cv_mean:.2f}+-{cv_std:.2f} <br>'

    def create_widget(self):
        # Select - Choose session to use for training
        self.widget_title = Div(text='<b>Select train ids </b>')
        self.select_session = CheckboxGroup()
        self.select_session.labels = self.available_sessions
        self.select_session.on_change('active', self.on_session_change)

        # Select - Choose model to train
        self.select_model = Select(title="Model")
        self.select_model.on_change('value', self.on_model_change)
        self.select_model.options = ['', 'CSP', 'FBCSP', 'Riemann']

        # Slider - Select ROI start (in s after start of epoch)
        self.slider_roi_start = Slider(start=0, end=6, value=2.5,
                                       step=0.25, title='ROI start (s)')

        # Slider - Select ROI end (in s after start of epoch)
        self.slider_roi_end = Slider(start=0, end=6, value=6,
                                     step=0.25, title='ROI end (s)')

        # Slider - Number of iterations if optimization
        self.slider_n_iters = Slider(start=1, end=50, value=1,
                                     title='Iterations (optimization)')

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference',
                                                            'Cropping'])

        self.checkbox_settings = CheckboxButtonGroup(
            labels=['Save', 'Optimize'])

        self.button_train = Button(label='Train', button_type='primary')
        self.button_train.on_click(self.on_train_start)

        self.div_info = Div()

        column1 = column(self.select_model, self.widget_title,
                         self.select_session)
        column2 = column(self.slider_roi_start, self.slider_roi_end,
                         self.checkbox_settings, self.slider_n_iters,
                         self.div_preproc, self.checkbox_preproc,
                         self.button_train, self.div_info)
        return row(column1, column2)
