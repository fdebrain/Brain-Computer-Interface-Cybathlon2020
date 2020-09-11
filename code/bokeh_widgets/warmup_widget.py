import logging
import time
from pathlib import Path
import traceback

import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models import Div, Select, CheckboxButtonGroup, Toggle
from pyqtgraph.Qt import QtCore

from src.lsl_client import LSLClient
from src.lsl_recorder import LSLRecorder
from src.action_predictor import ActionPredictor


class WarmUpWidget:
    def __init__(self, parent=None):
        self.fs = 500
        self.n_channels = 63
        self.t0 = 0

        # Chronogram
        self.chrono_source = ColumnDataSource(dict(ts=[], y_pred=[]))
        self.pred2encoding = {0: 'Rest', 1: 'Left', 2: 'Right', 3: 'Headlight'}

        # LSL stream reader
        self.parent = parent
        self.lsl_every_s = 0.1
        self.lsl_reader = None
        self.lsl_start_time = None
        self.thread_lsl = QtCore.QThreadPool()
        self.channel_source = ColumnDataSource(dict(ts=[], data=[]))
        self._lsl_data, self.lsl_ts = None, None

        # LSL stream recorder
        self.lsl_recorder = None
        self.h5_name = 'warmup_recording.h5'
        # self.thread_record = QtCore.QThreadPool()

        # Model
        self.models_path = Path('./saved_models')
        self.input_signal = np.zeros((self.n_channels, 4 * self.fs))
        self.n_crops = 10
        self.crop_len = 0.5
        self.predictor = None
        self.thread_pred = QtCore.QThreadPool()
        self._current_pred = (0, 'Rest')

        # Feedback images
        self.static_folder = Path('code/static')
        self.action2image = {'Left': 'arrow-left-solid.png',
                             'Right': 'arrow-right-solid.png',
                             'Rest': 'retweet-solid.png',
                             'Headlight': 'bolt-solid.png'}

    @property
    def current_pred(self):
        return self._current_pred

    @current_pred.setter
    def current_pred(self, action_idx):
        self._current_pred = (action_idx, self.pred2encoding[action_idx])
        self.parent.add_next_tick_callback(self.update_prediction)

    @property
    def lsl_data(self):
        return self._lsl_data

    @lsl_data.setter
    def lsl_data(self, data):
        self._lsl_data = data
        self.parent.add_next_tick_callback(self.update_signal)

    @property
    def lsl_ts(self):
        return self._lsl_ts

    @lsl_ts.setter
    def lsl_ts(self, ts):
        self._lsl_ts = ts

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
    def modelfile(self):
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

    def reset_predictor(self):
        self.model_info.text = f'<b>Model:</b> None'
        self.pred_info.text = f'<b>Prediction:</b> None'
        self.image.text = ''
        if self.predictor:
            self.predictor.should_predict = False
            self.predictor = None
            self.thread_pred.clear()

    def reset_lsl(self):
        if self.lsl_reader:
            self.lsl_reader.should_stream = False
            self.lsl_reader = None
            self.thread_lsl.clear()

    def on_settings_change(self, attr, old, new):
        self.plot_stream.visible = 0 in new

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models

        # Delete existing predictor thread
        if self.predictor is not None:
            logging.info(f'Remove old predictor thread')
            self.reset_predictor()

            if new == '':
                return

        try:
            self.predictor = ActionPredictor(self, self.modelfile, self.fs,
                                             predict_every_s=1)
            self.thread_pred.start(self.predictor)
            self.model_info.text = f'<b>Model:</b> {new}'
        except Exception:
            logging.error(f'Failed loading model {self.modelfile}')
            self.reset_predictor()

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data['data'] = []
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def reset_chronogram(self):
        logging.info('Reset chronogram')
        self.chrono_source.data = dict(ts=[], y_pred=[])

    def update_prediction(self):
        # Update chronogram source
        action_idx = self.current_pred[0]
        if self.lsl_start_time is not None:
            ts = time.time() - self.lsl_start_time
            self.chrono_source.stream(dict(ts=[ts],
                                           y_pred=[action_idx]))

        # Update information display (might cause )
        self.pred_info.text = f'<b>Prediction:</b> {self.current_pred}'
        src = self.static_folder / \
            self.action2image[self.pred2encoding[action_idx]]
        self.image.text = f"<img src={src} width='200' height='200' text-align='center'>"

    def on_lsl_connect_toggle(self, active):
        if active:
            # Connect to LSL stream
            self.button_lsl.label = 'Seaching...'
            self.button_lsl.button_type = 'warning'
            self.parent.add_next_tick_callback(self.on_lsl_connect)
        else:
            self.reset_lsl()
            self.reset_predictor()
            self.button_lsl.label = 'LSL Disconnected'
            self.button_lsl.button_type = 'danger'

    def on_lsl_connect(self):
        try:
            self.lsl_reader = LSLClient(self, fetch_every_s=self.lsl_every_s)
            self.fs = self.lsl_reader.fs

            if self.lsl_reader is not None:
                self.select_channel.options = [f'{i+1} - {ch}' for i, ch
                                               in enumerate(self.lsl_reader.ch_names)]
                self.thread_lsl.start(self.lsl_reader)
                self.button_lsl.label = 'Reading LSL stream'
                self.button_lsl.button_type = 'success'
        except Exception:
            logging.info(f'No LSL stream - {traceback.format_exc()}')
            self.button_lsl.label = 'Can\'t find stream'
            self.button_lsl.button_type = 'danger'
            self.reset_lsl()

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

    def update_signal(self):
        data, ts = self.lsl_data, self.lsl_ts

        # Truncate if different length
        len_ts = len(ts)
        len_data = data.shape[-1]
        if len_ts > len_data:
            logging.info(f'Truncate ts by {len_ts - len_data} values')
            ts = ts[:len_data]
        elif len_ts < len_data:
            logging.info(f'Truncate data by {len_data - len_ts} values')
            data = data[:, :len_ts]

        if data is None:
            logging.info('Data is empty')
            return

        if len(data.shape) == 1:
            logging.info('Skipping data points (bad format)')
            return

        # Convert timestamps in seconds
        if self.lsl_start_time is None:
            self.lsl_start_time = time.time()
            self.t0 = ts[0]
        ts -= self.t0

        # Update source display
        ch = self.channel_idx
        self.channel_source.stream(dict(ts=ts, data=data[ch, :]),
                                   rollover=int(2 * self.fs))

        # Update signal
        chunk_size = data.shape[-1]
        self.input_signal = np.roll(self.input_signal, -chunk_size, axis=-1)
        self.input_signal[:, -chunk_size:] = data

        # Record signal
        if self.lsl_recorder is not None:
            print('Recording')
            self.lsl_recorder.save_data(data, ts)

    def create_widget(self):
        # Toggle - Connect to LSL stream
        self.button_lsl = Toggle(label='Connect to LSL')
        self.button_lsl.on_click(self.on_lsl_connect_toggle)

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
        self.plot_chronogram.legend.background_fill_alpha = 0.6
        self.plot_chronogram.yaxis.ticker = list(self.pred2encoding.keys())
        self.plot_chronogram.yaxis.major_label_overrides = self.pred2encoding

        # Div - Display useful information
        self.model_info = Div(text=f'<b>Model:</b> None')
        self.pred_info = Div(text=f'<b>Prediction:</b> None')
        self.image = Div()

        # Create layout
        column1 = column(self.button_lsl,
                         self.button_record,
                         self.select_model,
                         self.select_channel,
                         self.div_settings, self.checkbox_settings,
                         self.div_preproc, self.checkbox_preproc)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.model_info, self.pred_info, self.image)
        return row(column1, column2, column3)
