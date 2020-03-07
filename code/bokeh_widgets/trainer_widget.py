
import logging
from pathlib import Path
import traceback

import numpy as np
from bokeh.io import curdoc
from bokeh.models import Div, Select, Button, Slider
from bokeh.models import CheckboxButtonGroup, CheckboxGroup
from bokeh.layouts import row, column

from src.dataloader import load_session, cropping, preprocessing
from src.models import train, save_model, save_json


class TrainerWidget:
    def __init__(self):
        self.data_path = Path('../Datasets/Pilots/Pilot_2')
        self.save_path = Path('./saved_models')

    @property
    def available_sessions(self):
        sessions = self.data_path.glob('*')
        return [s.name for s in sessions]

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
    def should_reref(self):
        return 'Rereference' in self.selected_preproc

    @property
    def should_filter(self):
        return 'Filter' in self.selected_preproc

    @property
    def should_standardize(self):
        return 'Standarsize' in self.selected_preproc

    @property
    def should_crop(self):
        return 'Crop' in self.selected_preproc

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
        X, y = {}, {}
        for id in self.train_ids:
            logging.info(f'Loading {id}')

            try:
                session_path = self.data_path / \
                    id / f'formatted_filt_500Hz'
                filepath = session_path / 'train/train1.npz'
                X[id], y[id], self.fs, self.ch_names = load_session(filepath,
                                                                    self.start,
                                                                    self.end)
            except Exception:
                logging.info(f'Loading data failed - {traceback.format_exc()}')
                self.button_train.button_type = 'danger'
                self.button_train.label = 'Training failed'
                return

        # Concatenate all data
        self.X = np.vstack([X[id] for id in X.keys()])
        self.y = np.hstack([y[id] for id in y.keys()]).flatten()
        logging.info(f'Shape: X {self.X.shape} - y {self.y.shape}')

        # Cropping
        if self.should_crop:
            self.X, self.y = cropping(self.X, self.y, self.fs,
                                      n_crops=8, crop_len=0.5)

        # Preprocessing
        self.X = preprocessing(self.X, self.fs,
                               rereference=self.should_reref,
                               filt=self.should_filter,
                               standardize=self.should_standardize)

        # Update session info
        self.div_info.text = f'<b>Sampling frequency:</b> {self.fs} Hz<br>' \
            f'<b>Classes:</b> {np.unique(self.y)} <br>' \
            f'<b>Nb trials:</b> {len(self.y)} <br>' \
            f'<b>Nb channels:</b> {self.X.shape[1]} <br>' \
            f'<b>Trial length:</b> {self.X.shape[-1] / self.fs}s <br>'

        self.button_train.label = 'Training...'
        curdoc().add_next_tick_callback(self.on_train)

    def on_train(self):
        try:
            trained_model, cv_mean, cv_std, train_time = train(self.model_name,
                                                               self.X, self.y,
                                                               self.train_mode,
                                                               n_iters=self.slider_n_iters)
        except Exception:
            logging.info(f'Training failed - {traceback.format_exc()}')
            self.button_train.button_type = 'danger'
            self.button_train.label = 'Failed'
            return

        logging.info(f'Trained successfully in {train_time:.0f}s \n'
                     f'Accuracy: {cv_mean:.2f}+-{cv_std:.2f} \n'
                     f'{trained_model}')

        model_to_save = trained_model if self.train_mode == 'validate' \
            else trained_model.best_estimator_

        if 'Save' in self.selected_settings:
            dataset_name = '_'.join([id for id in self.train_ids])
            filename = f'{self.model_name}_{dataset_name}'
            save_model(model_to_save, self.save_path, filename)

            model_info = {"Model name": self.model_name,
                          "Model file": filename,
                          "Train ids": self.train_ids,
                          "fs": int(self.fs),
                          "Shape": self.X.shape,
                          "Preprocessing": self.selected_preproc,
                          "Model pipeline": {k: str(v) for k, v in model_to_save.steps},
                          "CV RMSE": f'{cv_mean:.3f}+-{cv_std:.3f}',
                          "Train time": train_time}
            save_json(model_info, self.save_path, filename)

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
        self.slider_roi_start = Slider(start=0, end=6, value=2,
                                       step=0.25, title='ROI start (s)')

        # Slider - Select ROI end (in s after start of epoch)
        self.slider_roi_end = Slider(start=0, end=6, value=6,
                                     step=0.25, title='ROI end (s)')

        self.checkbox_settings = CheckboxButtonGroup(labels=['Save',
                                                             'Optimize'])

        # Slider - Number of iterations if optimization
        self.slider_n_iters = Slider(start=1, end=50, value=1,
                                     title='Iterations (optimization)')

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference',
                                                            'Crop'])

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
