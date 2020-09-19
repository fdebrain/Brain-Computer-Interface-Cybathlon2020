import os
import logging
import time
import traceback
import copy

import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models import Div, Select, CheckboxButtonGroup, Toggle
from pyqtgraph.Qt import QtCore
import mne

from config import main_config, warmup_config
from src.lsl_client import LSLClient
from src.lsl_recorder import LSLRecorder
from src.action_predictor import ActionPredictor


class WarmUpWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.fs = main_config['fs']
        self.n_channels = main_config['n_channels']
        self.t0 = 0
        self.last_ts = 0

        # Chronogram
        self.chrono_source = ColumnDataSource(dict(ts=[], y_pred=[]))
        self.pred_decoding = main_config['pred_decoding']

        # LSL stream reader
        self.lsl_reader = None
        self.lsl_start_time = None
        self._lsl_data = (None, None)
        self.thread_lsl = QtCore.QThreadPool()
        self.channel_source = ColumnDataSource(dict(ts=[], eeg=[]))
        self.buffer_size_s = 10

        # LSL stream recorder
        if not os.path.isdir(main_config['record_path']):
            os.mkdir(main_config['record_path'])
        self.record_path = main_config['record_path']
        self.record_name = warmup_config['record_name']
        self.lsl_recorder = None

        # Predictor
        self.models_path = main_config['models_path']
        self.input_signal = np.zeros((self.n_channels, 4 * self.fs))
        self.predictor = None
        self.thread_pred = QtCore.QThreadPool()
        self._current_pred = (0, 'Rest')

        # Feedback images
        self.static_folder = warmup_config['static_folder']
        self.action2image = warmup_config['action2image']

        # Signal preprocessing
        self.apply_notch = warmup_config['apply_notch']
        self.apply_filt = warmup_config['apply_filt']
        self.f_min = warmup_config['f_min']
        self.f_max = warmup_config['f_max']

    @property
    def current_pred(self):
        return self._current_pred

    @current_pred.setter
    def current_pred(self, action_idx):
        self._current_pred = (action_idx, self.pred_decoding[action_idx])
        self.parent.add_next_tick_callback(self.update_prediction)

    @property
    def lsl_data(self):
        return self._lsl_data

    @lsl_data.setter
    def lsl_data(self, data):
        self._lsl_data = data
        self.parent.add_next_tick_callback(self.update_signal)

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
    def is_convnet(self):
        return self.select_model.value.split('.')[-1] == 'h5'

    @property
    def channel_idx(self):
        return int(self.select_channel.value.split('-')[0])

    def reset_lsl(self):
        if self.lsl_reader:
            self.lsl_reader.should_stream = False
            self.lsl_reader = None
            self.lsl_start_time = None
            self.thread_lsl.clear()

    def reset_predictor(self):
        self.model_info.text = f'<b>Model:</b> None'
        self.pred_info.text = f'<b>Prediction:</b> None'
        self.image.text = ''
        if self.predictor:
            self.predictor.should_predict = False
            self.predictor = None
            self.thread_pred.clear()

    def reset_recorder(self):
        if self.lsl_recorder:
            self.lsl_recorder.close_h5()
            self.lsl_recorder = None

    def on_settings_change(self, attr, old, new):
        self.plot_stream.visible = 0 in new

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models

        # Delete existing predictor thread
        if self.predictor is not None:
            self.reset_predictor()
            if new == '':
                return

        try:
            self.predictor = ActionPredictor(self,
                                             self.modelfile,
                                             self.is_convnet)
            self.thread_pred.start(self.predictor)
            self.model_info.text = f'<b>Model:</b> {new}'
        except Exception as e:
            logging.error(f'Failed loading model {self.modelfile} - {e}')
            self.reset_predictor()

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data = dict(ts=[], eeg=[])
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def reset_plots(self):
        self.chrono_source.data = dict(ts=[], y_pred=[])
        self.channel_source.data = dict(ts=[], eeg=[])

    def update_prediction(self):
        # Update chronogram source
        action_idx = self.current_pred[0]
        if self.lsl_start_time is not None:
            ts = time.time() - self.lsl_start_time
            self.chrono_source.stream(dict(ts=[ts],
                                           y_pred=[action_idx]))

        # Update information display (might cause delay)
        self.pred_info.text = f'<b>Prediction:</b> {self.current_pred}'
        src = self.static_folder / \
            self.action2image[self.pred_decoding[action_idx]]
        self.image.text = f"<img src={src} width='200' height='200' text-align='center'>"

        # Save prediction as event
        if self.lsl_recorder is not None:
            self.lsl_recorder.save_event(copy.deepcopy(self.last_ts),
                                         copy.deepcopy(action_idx))

    def on_lsl_connect_toggle(self, active):
        if active:
            # Connect to LSL stream
            self.button_lsl.label = 'Seaching...'
            self.button_lsl.button_type = 'warning'
            self.reset_plots()
            self.parent.add_next_tick_callback(self.start_lsl_thread)
        else:
            self.reset_lsl()
            self.reset_predictor()
            self.button_lsl.label = 'LSL Disconnected'
            self.button_lsl.button_type = 'danger'

    def start_lsl_thread(self):
        try:
            self.lsl_reader = LSLClient(self)
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
                self.lsl_recorder = LSLRecorder(self.record_path,
                                                self.record_name)
                self.lsl_recorder.open_h5()
            except Exception as e:
                logging.info(f'Failed creating LSLRecorder - {e}')
                self.reset_recorder()
                self.button_record.label = 'Recording failed'
                self.button_record.button_type = 'danger'

            self.button_record.label = 'Stop recording'
            self.button_record.button_type = 'success'
        else:
            self.reset_recorder()
            self.button_record.label = 'Start recording'
            self.button_record.button_type = 'primary'

    def preproc_signal(self, eeg):
        info = mne.create_info(self.lsl_reader.ch_names,
                               self.fs,
                               ch_types='eeg')
        data = mne.io.RawArray(eeg, info, verbose=0)

        if self.apply_notch:
            data.notch_filter(freqs=[50], filter_length=40)
        if self.apply_filt:
            # data.savgol_filter(h_freq=30, verbose=0)
            data.filter(l_freq=self.f_min, h_freq=self.f_max)

        return data

    def update_signal(self):
        ts, eeg = self.lsl_data
        self.last_ts = ts[-1]

        if ts.shape[0] != eeg.shape[-1]:
            logging.info('Skipping data points (bad format)')
            return

        # Convert timestamps in seconds
        if self.lsl_start_time is None:
            self.lsl_start_time = time.time()
            self.t0 = ts[0]

        # Filter channel signal for visualization purpose
        ch = self.channel_idx
        data = self.preproc_signal(eeg)
        channel_eeg, _ = data[ch, :]

        # Update source display
        self.channel_source.stream(dict(ts=ts-self.t0, eeg=channel_eeg.flatten()),
                                   rollover=int(self.buffer_size_s * self.fs))

        # Update signal
        chunk_size = eeg.shape[-1]
        self.input_signal = np.roll(self.input_signal, -chunk_size, axis=-1)
        self.input_signal[:, -chunk_size:] = eeg

        # Record signal
        if self.lsl_recorder is not None:
            self.lsl_recorder.save_data(copy.deepcopy(ts), copy.deepcopy(eeg))

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
        self.plot_stream.line(x='ts', y='eeg', source=self.channel_source)

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
        self.plot_chronogram.yaxis.ticker = list(self.pred_decoding.keys())
        self.plot_chronogram.yaxis.major_label_overrides = self.pred_decoding

        # Div - Display useful information
        self.model_info = Div(text=f'<b>Model:</b> None')
        self.pred_info = Div(text=f'<b>Prediction:</b> None')
        self.image = Div()

        # Create layout
        column1 = column(self.button_lsl,
                         self.button_record,
                         self.select_model,
                         self.select_channel,
                         self.div_settings, self.checkbox_settings)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.model_info, self.pred_info, self.image)
        return row(column1, column2, column3)
