import numpy as np
import logging
import glob
import os
from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, Button, CheckboxButtonGroup, CheckboxGroup
from bokeh.layouts import widgetbox
from data_loading_functions.data_loader import EEGDataset
from feature_extraction_functions.models import train, save_model


class TrainerWidget:
    def __init__(self):
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')
        self.X, self.y = None, None
        self.fs = 500

    @property
    def model_name(self):
        return self.select_model.value

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

    def create_widget(self):
        self.select_model = Select(title="Model")
        self.select_model.on_change('value', self.on_model_change)
        self.select_model.options = ['', 'CSP', 'FBCSP',
                                     'Riemann', 'ShallowConv']

        # Select - Choose session to use for training
        self.widget_title = Div(text='<b>Select train ids </b>')
        self.select_session = CheckboxGroup()
        self.select_session.labels = self.available_sessions
        self.select_session.on_change('active', self.on_session_change)

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference'])

        self.checkbox_settings = CheckboxButtonGroup(labels=['Save'])

        self.button_train = Button(label='Train', button_type='primary')
        self.button_train.on_click(self.on_train_start)

        self.div_info = Div()

        layout = widgetbox([self.select_model,
                            self.widget_title, self.select_session,
                            self.checkbox_settings,
                            self.div_preproc, self.checkbox_preproc,
                            self.button_train, self.div_info])
        return layout

    def on_model_change(self, attr, old, new):
        logging.info(f'Select model {new}')
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'
        self.div_info.text = ''

    def on_session_change(self, attr, old, new):
        logging.info(f"Select train sessions {new}")
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'

    def on_train_start(self):
        assert self.model_name != '', 'Please select a model !'
        assert len(self.train_ids) > 0, 'Please select at least one session !'

        self.button_train.button_type = 'warning'
        self.button_train.label = 'Loading data...'
        curdoc().add_next_tick_callback(self.on_load)

    def on_load(self):
        X, y = {}, {}
        for session_path in self.train_ids:
            id = session_path.split('/')[-1]
            logging.info(f'Loading {id}')
            self.dataloader_params = {
                'data_path': f'{session_path}/formatted_filt_{self.fs}Hz/',
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
                self.button_train.label = 'Failed'
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
        # Extract MI phase TODO: Add sliders widgets
        fs = self.fs
        start = 2.5
        end = 6.
        logging.info(f'Extracting MI: [{start} to {end}]s')
        self.X = self.X[:, :, int(start*fs): int(end*fs)]

        # Training model TODO: Add n_iters slider
        try:
            cv_model, acc, std, timing = train(self.model_name,
                                               self.X, self.y,
                                               n_iters=10)
        except Exception as e:
            logging.info(f'Training failed - {e}')
            self.button_train.button_type = 'danger'
            self.button_train.label = 'Failed'
            return

        logging.info(f'Trained successfully in {timing:.0f}s \n'
                     f'Accuracy: {acc:.2f}+-{std:.2f} \n'
                     f'{cv_model.best_estimator_}')

        if 'Save' in self.selected_settings:
            dir_path = './saved_models'
            if not os.path.isdir(dir_path):
                logging.info(f'Creating directory {dir_path}')
                os.mkdir(dir_path)

            logging.info(f'Saving model...')
            dataset_name = '_'.join([id.split('/')[-1]
                                     for id in self.train_ids])
            pkl_filename = f"{self.model_name}_{dataset_name}.pkl"
            save_model(cv_model.best_estimator_, dir_path, pkl_filename)
            logging.info('Successfully saved model !')

        # Update info
        self.button_train.button_type = 'success'
        self.button_train.label = 'Trained'
        self.div_info.text += f'<b>Accuracy:</b> {acc:.2f}+-{std:.2f} <br>'
