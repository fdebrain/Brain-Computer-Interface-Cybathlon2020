import logging
import time
from pathlib import Path
import traceback

import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models import Button, Div, Select, CheckboxButtonGroup
from pyqtgraph.Qt import QtCore

from src.pipeline import load_pipeline
from src.preprocessing import cropping, preprocessing
from src.models import predict
from src.lsl_client import LSLClient
from src.lsl_recorder import LSLRecorder


class WarmUpWidget:
    def __init__(self, parent=None):
        self.fs = 500
        self.n_channels = 61

        # Chronogram
        self.callback_action_id = None
        self.chrono_source = ColumnDataSource(dict(ts=[], y_pred=[]))
        self.pred2encoding = {0: 'Rest', 1: 'Left', 2: 'Right', 3: 'Headlight'}

        # LSL stream reader
        self.parent = parent
        self.lsl_reader = None
        self.lsl_start_time = None
        self.thread_lsl = QtCore.QThreadPool()
        self.callback_lsl_id = None
        self.channel_source = ColumnDataSource(dict(ts=[], data=[]))

        # Model
        self.models_path = Path('./saved_models')
        self.model = None
        self.signal = None
        self.current_pred = (0, 'Rest')

        # Feedback images
        self.static_folder = Path('code/static')
        self.action2image = {'Left': 'arrow-left-solid.png',
                             'Right': 'arrow-right-solid.png',
                             'Rest': 'retweet-solid.png',
                             'Headlight': 'bolt-solid.png'}

    @property
    def available_models(self):
        ml_models = [p.name for p in self.models_path.glob('*.pkl')]
        dl_models = [p.name for p in self.models_path.glob('*.h5')]
        return [''] + ml_models + dl_models

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def model_path(self):
        return self.models_path / self.select_model.value

    @property
    def model_name(self):
        return self.select_model.value

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
        return 'Standardize' in self.selected_preproc

    @property
    def is_convnet(self):
        return self.select_model.value.split('.')[-1] == 'h5'

    @property
    def channel_idx(self):
        return int(self.select_channel.value.split('-')[0])

    def on_settings_change(self, attr, old, new):
        self.plot_stream.visible = 0 in new

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models
        self.model = load_pipeline(self.model_path)

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data['data'] = []
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def clear_chronogram(self):
        logging.info('Clear chronogram')
        self.chrono_source.data = dict(ts=[], y_pred=[])
        self.div_info.text = ''

    def create_action_callback(self):
        if self.callback_action_id is None:
            logging.info('Create action callback')
            self.callback_action_id = self.parent.add_periodic_callback(
                self.callback_action, 1000)

    def remove_action_callback(self):
        if self.callback_action_id:
            logging.info('Remove action callback')
            self.parent.remove_periodic_callback(self.callback_action_id)
            self.callback_action_id = None

    def callback_action(self):
        ''' This periodic callback starts at the same time as the race '''

        assert self.callback_lsl_id is not None, 'Please connect to LSL stream'
        model_name = self.model_name

        X = np.copy(self.signal)

        # Removing FP1 & FP2 TODO: Don't hardcode
        X = np.delete(X, [0, 30], axis=0)

        # Selecting last 1s of signal
        X = X[np.newaxis, :, -self.fs:]

        # Preprocess
        X = preprocessing(X, self.fs, self.should_reref,
                          self.should_filter,
                          self.should_standardize)

        # Cropping
        X, _ = cropping(X, [None], self.fs,
                        n_crops=10, crop_len=0.5)

        # If no model
        if self.model is None:
            action_idx = 0
            logging.warning('Rest action sent by default!'
                            'Please select a model.')
        else:
            action_idx = predict(X, self.model, self.is_convnet)
            logging.info(f'Action idx: {action_idx}')

        self.current_pred = (action_idx, self.pred2encoding[action_idx])

        # Update chronogram source (if race started)
        if self.lsl_start_time is not None:
            ts = time.time() - self.lsl_start_time
            self.chrono_source.stream(dict(ts=[ts],
                                           y_pred=[action_idx]))

        # Update information display
        self.div_info.text = f'<b>Model:</b> {model_name} <br>' \
            f'<b>Prediction:</b> {self.current_pred} <br>' \

        src = self.static_folder / \
            self.action2image[self.pred2encoding[action_idx]]
        self.image.text = f"<img src={src} width='200' height='200' text-align='center'>"

    def on_lsl_connect_start(self):
        if self.lsl_reader is not None:
            logging.info('Delete old lsl stream')
            self.thread_lsl.clear()
            self.lsl_reader = None

        self.button_lsl.label = 'Seaching...'
        self.button_lsl.button_type = 'warning'
        self.parent.add_next_tick_callback(self.on_lsl_connect)

    def on_lsl_connect(self):
        try:
            self.lsl_reader = LSLClient()
            self.fs = self.lsl_reader.fs
            self.signal = np.zeros((self.lsl_reader.n_channels, 4 * self.fs))

            if self.lsl_reader is not None:
                logging.info('Start periodic callback - LSL')
                self.select_channel.options = [f'{i+1} - {ch}' for i, ch
                                               in enumerate(self.lsl_reader.ch_names)]
                self.create_lsl_callback()
                self.button_lsl.label = 'Reading LSL stream'
                self.button_lsl.button_type = 'success'

                self.parent.add_next_tick_callback(self.create_action_callback)

        except Exception:
            logging.info(f'No LSL stream - {traceback.format_exc()}')
            self.button_lsl.label = 'Can\'t find stream'
            self.button_lsl.button_type = 'danger'
    def on_lsl_record_toggle(self, active):
        if active:
            try:
                self.lsl_recorder = LSLRecorder(self.h5_name,
                                                self.n_channels,
                                                self.fs)
            except Exception:
                self.reset_lsl()
                self.button_record.label = 'Recording failed'
                self.button_record.button_type = 'danger'

            self.button_record.label = 'Stop recording'
            self.button_record.button_type = 'success'
        else:
            self.reset_lsl()
            self.button_record.label = 'Start recording'
            self.button_record.button_type = 'primary'

    def create_lsl_callback(self):
        if self.callback_lsl_id is None:
            logging.info('Create LSL callback')
            self.callback_lsl_id = self.parent.add_periodic_callback(
                self.callback_lsl, 100)

    def remove_lsl_callback(self):
        if self.callback_lsl_id:
            logging.info('Remove LSL callback')
            self.parent.remove_periodic_callback(self.callback_lsl_id)
            self.callback_lsl_id = None

    def _fetch_data(self):
        data, ts = self.lsl_reader.get_data()

        if len(data.shape) == 1:
            logging.info('Skipping data points (bad format)')
            return

        # Convert timestamps in seconds
        if self.lsl_start_time is None:
            self.lsl_start_time = time.time()
        ts -= self.lsl_start_time

        # Clean signal and reference
        data = np.swapaxes(data, 1, 0)

        # Update source
        ch = self.channel_idx
        self.channel_source.stream(dict(ts=ts, data=data[ch, :]),
                                   rollover=int(2 * self.fs))

        # Update signal
        chunk_size = data.shape[-1]
        self.signal = np.roll(self.signal, -chunk_size, axis=-1)
        self.signal[:, -chunk_size:] = data

    def callback_lsl(self):
        ''' Fetch EEG data from LSL stream '''
        try:
            self._fetch_data()
        except Exception:
            logging.info(
                f'Ending periodic callback - {traceback.format_exc()}')
            self.remove_lsl_callback()
            self.remove_action_callback()
            self.button_lsl.label = 'No stream'
            self.button_lsl.button_type = 'danger'

    def create_widget(self):
        # Toggle - Start/stop LSL stream recording
        self.button_record = Toggle(label='Start Recording',
                                    button_type='primary')
        self.button_record.on_click(self.on_lsl_record_toggle)

        # Select - Choose pre-trained model
        self.select_model = Select(title="Select pre-trained model")
        self.select_model.options = self.available_models
        self.select_model.on_change('value', self.on_model_change)

        # Checkbox - Choose settings
        self.div_settings = Div(text='<b>Settings</b>', align='center')
        self.checkbox_settings = CheckboxButtonGroup(labels=['Show signal'])
        self.checkbox_settings.on_change('active', self.on_settings_change)

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference'],
                                                    active=[1, 2])

        # Select - Channel to visualize
        self.select_channel = Select(title='Select channel', value='1 - Fp1')
        self.select_channel.on_change('value', self.on_channel_change)

        # Plot - LSL EEG Stream
        self.plot_stream = figure(title='Temporal EEG signal',
                                  x_axis_label='Time [s]',
                                  y_axis_label='Amplitude',
                                  plot_height=500,
                                  plot_width=800,
                                  visible=False)
        self.plot_stream.line(x='ts', y='data', source=self.channel_source)

        # Plot - Chronogram prediction vs results
        self.plot_chronogram = figure(title='Chronogram',
                                      x_axis_label='Time [s]',
                                      y_axis_label='Action',
                                      plot_height=300,
                                      plot_width=800)
        self.plot_chronogram.cross(x='ts', y='y_pred', color='red',
                                   source=self.chrono_source,
                                   legend_label='Prediction')
        self.plot_chronogram.legend.background_fill_alpha = 0.8
        self.plot_chronogram.yaxis.ticker = list(self.pred2encoding.keys())
        self.plot_chronogram.yaxis.major_label_overrides = self.pred2encoding

        # Div - Display useful information
        self.div_info = Div()
        self.image = Div()

        # Create layout
        column1 = column(self.button_lsl,
                         self.button_record,
                         self.select_model,
                         self.select_channel,
                         self.div_settings, self.checkbox_settings,
                         self.div_preproc, self.checkbox_preproc)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.div_info, self.image)
        return row(column1, column2, column3)
