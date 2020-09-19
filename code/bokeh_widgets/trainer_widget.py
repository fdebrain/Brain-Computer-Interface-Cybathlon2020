
import logging
import traceback

import numpy as np
from bokeh.io import curdoc
from bokeh.models import Div, Select, Button, Slider
from bokeh.models import CheckboxButtonGroup, MultiChoice
from bokeh.layouts import row, column

from src.dataloader import load_session
from src.preprocessing import cropping
from src.pipeline import get_pipeline, save_json, save_pipeline
from src.trainer import train

from config import main_config


class TrainerWidget:
    def __init__(self):
        self.data_path = main_config['data_path']
        self.save_path = main_config['models_path']
        self.active_preproc_ordered = []

    @property
    def available_pilots(self):
        pilots = self.data_path.glob('*')
        return [''] + [p.parts[-1] for p in pilots]

    @property
    def selected_pilot(self):
        return self.select_pilot.value

    @property
    def available_sessions(self):
        pilot_path = self.data_path / self.selected_pilot
        sessions = pilot_path.glob('*')
        return [s.name for s in sessions]

    @property
    def selected_preproc(self):
        active = self.active_preproc_ordered
        return [self.checkbox_preproc.labels[i] for i in active]

    @property
    def train_ids(self):
        return self.select_session.value

    @property
    def preproc_config(self):
        config_cn = dict(sigma=6)
        config_bpf = dict(fs=self.fs, f_order=2, f_type='butter',
                          f_low=4, f_high=38)
        config_crop = dict(fs=self.fs, n_crops=10, crop_len=0.5)
        return {'CN': config_cn, 'BPF': config_bpf, 'Crop': config_crop}

    @property
    def should_crop(self):
        return 'Crop' in self.selected_preproc

    @property
    def selected_folders(self):
        active = self.checkbox_folder.active
        return [self.checkbox_folder.labels[i] for i in active]

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def model_name(self):
        return self.select_model.value

    @property
    def model_config(self):
        config = {'model_name': self.model_name,
                  'C': 10}
        return config

    @property
    def is_convnet(self):
        return self.model_name == 'ConvNet'

    @property
    def train_mode(self):
        return 'optimize' if 'Optimize' in self.selected_settings \
            else 'validate'

    @property
    def folder_ids(self):
        ids = []
        if 'New Calib' in self.selected_folders:
            ids.append('formatted_filt_500Hz')
        if 'Game' in self.selected_folders:
            ids.append('formatted_filt_500Hz_game')
        return ids

    @property
    def start(self):
        return self.slider_roi_start.value

    @property
    def end(self):
        return self.slider_roi_end.value

    @property
    def n_iters(self):
        return self.slider_n_iters.value

    def on_pilot_change(self, attr, old, new):
        logging.info(f'Select pilot {new}')
        self.select_session.value = ['']
        self.update_widget()

    def on_session_change(self, attr, old, new):
        logging.info(f"Select train sessions {new}")
        self.update_widget()

    def on_model_change(self, attr, old, new):
        logging.info(f'Select model {new}')
        self.update_widget()

    def on_preproc_change(self, attr, old, new):
        # Case 1: Add preproc
        if len(new) > len(old):
            to_add = list(set(new) - set(old))[0]
            self.active_preproc_ordered.append(to_add)
        # Case 2: Remove preproc
        else:
            to_remove = list(set(old) - set(new))[0]
            self.active_preproc_ordered.remove(to_remove)

        logging.info(f'Preprocessing selected: {self.selected_preproc}')
        self.update_widget()

    def update_widget(self):
        self.select_pilot.options = self.available_pilots
        self.select_session.options = self.available_sessions
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'
        self.div_info.text = f'<b>Preprocessing selected:</b> {self.selected_preproc} <br>'

    def on_train_start(self):
        assert self.model_name != '', 'Please select a model !'
        assert len(self.train_ids) > 0, 'Please select at least one session !'

        self.button_train.button_type = 'warning'
        self.button_train.label = 'Loading data...'
        curdoc().add_next_tick_callback(self.on_load)

    def on_load(self):
        X, y = {}, {}
        for id in self.train_ids:
            for folder in self.folder_ids:
                logging.info(f'Loading {id} - {folder}')
                try:
                    session_path = self.data_path / self.selected_pilot /\
                        id / folder
                    filepath = session_path / 'train/train1.npz'
                    X_id, y_id, fs, ch_names = load_session(filepath,
                                                            self.start,
                                                            self.end)
                    X[f'{id}_{folder}'] = X_id
                    y[f'{id}_{folder}'] = y_id
                    self.fs = fs
                    self.ch_names = ch_names

                except Exception as e:
                    logging.info(f'Loading data failed - {e}')
                    self.button_train.button_type = 'danger'
                    self.button_train.label = 'Training failed'
                    return

        # Concatenate all data
        self.X = np.vstack([X[id] for id in X.keys()])
        self.y = np.hstack([y[id] for id in y.keys()]).flatten()

        # Cropping
        if self.should_crop:
            self.X, self.y = cropping(self.X, self.y,
                                      **self.preproc_config['Crop'])

        if self.is_convnet:
            assert self.should_crop, 'ConvNet requires cropping !'
            self.X = self.X[:, :, :, np.newaxis]

        # Update session info
        self.div_info.text = f'<b>Sampling frequency:</b> {self.fs} Hz<br>' \
            f'<b>Classes:</b> {np.unique(self.y)} <br>' \
            f'<b>Nb trials:</b> {len(self.y)} <br>' \
            f'<b>Nb channels:</b> {self.X.shape[1]} <br>' \
            f'<b>Trial length:</b> {self.X.shape[-1] / self.fs}s <br>'

        self.button_train.label = 'Training...'
        curdoc().add_next_tick_callback(self.on_train)

    def on_train(self):
        pipeline, search_space = get_pipeline(self.selected_preproc,
                                              self.preproc_config,
                                              self.model_config)

        try:
            logging.info(f'Shape: X {self.X.shape} - y {self.y.shape}')
            trained_model, cv_mean, cv_std, train_time = train(self.X, self.y,
                                                               pipeline,
                                                               search_space,
                                                               self.train_mode,
                                                               self.n_iters,
                                                               n_jobs=-1,
                                                               is_convnet=self.is_convnet)
        except Exception:
            logging.info(f'Training failed - {traceback.format_exc()}')
            self.button_train.button_type = 'danger'
            self.button_train.label = 'Failed'
            return

        model_to_save = trained_model if self.train_mode == 'validate' \
            else trained_model.best_estimator_

        if 'Save' in self.selected_settings:
            dataset_name = '_'.join([id for id in self.train_ids])
            filename = f'{self.model_name}_{dataset_name}'
            save_pipeline(model_to_save, self.save_path, filename)

            model_info = {"Model name": self.model_name,
                          "Model file": filename,
                          "Train ids": self.train_ids,
                          "fs": self.fs,
                          "Shape": self.X.shape,
                          "Preprocessing": self.selected_preproc,
                          "Model pipeline": {k: str(v) for k, v in model_to_save.steps},
                          "CV RMSE": f'{cv_mean:.3f}+-{cv_std:.3f}',
                          "Train time": train_time}
            save_json(model_info, self.save_path, filename)

        logging.info(f'{model_to_save} \n'
                     f'Trained successfully in {train_time:.0f}s \n'
                     f'Accuracy: {cv_mean:.2f}+-{cv_std:.2f}')

        # Update info
        self.button_train.button_type = 'success'
        self.button_train.label = 'Trained'
        self.div_info.text += f'<b>Accuracy:</b> {cv_mean:.2f}+-{cv_std:.2f} <br>'

    def create_widget(self):
        # Select - Pilot
        self.select_pilot = Select(title='Pilot:',
                                   options=self.available_pilots)
        self.select_pilot.on_change('value', self.on_pilot_change)

        # Multichoice - Choose training folder
        self.checkbox_folder = CheckboxButtonGroup(labels=['New Calib',
                                                           'Game'])

        # Multichoice - Choose session to use for training
        self.select_session = MultiChoice(title='Select train ids',
                                          width=250, height=120)
        self.select_session.on_change('value', self.on_session_change)

        # Select - Choose model to train
        self.select_model = Select(title="Model")
        self.select_model.on_change('value', self.on_model_change)
        self.select_model.options = ['', 'CSP', 'FBCSP', 'Riemann', 'ConvNet']

        # Slider - Select ROI start (in s after start of epoch)
        self.slider_roi_start = Slider(start=0, end=6, value=2,
                                       step=0.25, title='ROI start (s)')

        # Slider - Select ROI end (in s after start of epoch)
        self.slider_roi_end = Slider(start=0, end=6, value=6,
                                     step=0.25, title='ROI end (s)')

        self.checkbox_settings = CheckboxButtonGroup(labels=['Save',
                                                             'Optimize'])

        # Slider - Number of iterations if optimization
        self.slider_n_iters = Slider(start=1, end=50, value=5,
                                     title='Iterations (optimization)')

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['BPF',
                                                            'CN',
                                                            'CAR',
                                                            'Crop'])
        self.checkbox_preproc.on_change('active', self.on_preproc_change)

        self.button_train = Button(label='Train', button_type='primary')
        self.button_train.on_click(self.on_train_start)

        self.div_info = Div()

        column1 = column(self.select_pilot, self.checkbox_folder,
                         self.select_session, self.select_model)
        column2 = column(self.slider_roi_start, self.slider_roi_end,
                         self.checkbox_settings, self.slider_n_iters,
                         self.div_preproc, self.checkbox_preproc,
                         self.button_train, self.div_info)
        return row(column1, column2)
