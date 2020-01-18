import numpy as np
import logging
import glob
from bokeh.models.widgets import Div, Select, Button
from bokeh.layouts import widgetbox
from data_loading_functions.data_loader import EEGDataset
from feature_extraction_functions.models import train

# TODO: Train & save model as pickle + Load trained model


class ModelWidget():
    def __init__(self):
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')
        self.X, self.y = None, None

    @property
    def model_name(self):
        return self.select_model.value

    @property
    def session_train(self):
        return self.select_session.value

    def create_widget(self):
        self.widget_title = Div(text='<b>Trainer</b>', align='center')

        self.select_model = Select(title="Model")
        self.select_model.on_change('value', self.on_model_change)
        self.select_model.options = ['', 'CSP', 'FBCSP',
                                     'Riemann', 'ShallowConv']

        self.select_session = Select(title="Session")
        self.select_session.on_change('value', self.on_session_change)
        self.select_session.options = [''] + self.available_sessions

        self.button_train = Button(label='Train', button_type='primary')
        self.button_train.on_click(self.on_train)

        self.info = Div()

        layout = widgetbox([self.widget_title, self.select_model,
                            self.select_session, self.button_train,
                            self.info])
        return layout

    def on_model_change(self, attr, old, new):
        logging.info(f'Select model {new}')
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'
        self.info.text = ''

    def on_session_change(self, attr, old, new):
        logging.info(f"Select session {new.split('_')[-1]}")
        self.button_train.button_type = 'primary'
        self.button_train.label = 'Train'

        # Loading data
        fs = 250
        self.dataloader_params = {
            'data_path': f'{self.session_train}/formatted_filt_250Hz/',
            'fs': fs,
            'filt': False,
            'rereferencing': False,
            'standardization': False,
            'valid_ratio': 0.0,
            'load_test': False,
            'balanced': True}
        dataset = EEGDataset(pilot_idx=1, **self.dataloader_params,)
        self.X, self.y, _, _, _, _ = dataset.load_dataset()

        # Update session info
        self.info.text = f'<b>Classes:</b> {np.unique(self.y)} <br>'
        self.info.text += f'<b>Nb trials:</b> {len(self.y)} <br>'
        self.info.text += f'<b>Nb channels:</b> {self.X.shape[1]} <br>'
        self.info.text += f'<b>Trial length:</b> {self.X.shape[-1] / fs}s <br>'

    def on_train(self):
        assert self.model_name != '', 'Please select a model !'
        assert self.session_train != '', 'Please select a session !'
        # Training model
        cv_model, acc, std = train(self.model_name, self.X, self.y)
        logging.info(f'Trained successfully \n'
                     f'Accuracy: {acc}+-{std} \n'
                     f'{cv_model.best_estimator_}')

        # Save trained model
        self.button_train.button_type = 'success'
        self.button_train.label = 'Trained'

        # Update info
        self.info.text += f'<b>Accuracy:</b> {acc:.2f}+-{std:.2f} <br>'
