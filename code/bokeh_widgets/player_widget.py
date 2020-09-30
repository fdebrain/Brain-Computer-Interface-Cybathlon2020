import os
import logging
import copy
import subprocess
import sys
import time
import traceback

import numpy as np
import serial
import serial.tools.list_ports
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, Button, Toggle, Div, Select,
                          CheckboxButtonGroup)
from pyqtgraph.Qt import QtCore
from sklearn.metrics import accuracy_score

from config import main_config, game_config
from src.lsl_client import LSLClient
from src.lsl_recorder import LSLRecorder
from src.action_predictor import ActionPredictor
from src.game_player import CommandSenderPort
from src.game_log_reader import GameLogReader
from .utils import clean_log_directory


class PlayerWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.fs = main_config['fs']
        self.n_channels = main_config['n_channels']
        self.t0 = 0
        self.last_ts = 0

        # Game log reader (separate thread)
        self.game_logs_path = game_config['game_logs_path']
        self.game_log_reader = None
        self._expected_action = (0, 'Rest')
        self.thread_log = QtCore.QThreadPool()

        # Game window (separate process)
        clean_log_directory(self.game_logs_path)
        self.game = None

        # Port event sender
        self.micro_path = main_config['micro_path']
        self.port_sender = None

        # Game player
        self.game_path = game_config['game_path']
        self.player_idx = game_config['player_idx']
        self.game_start_time = None

        # Chronogram
        self.chrono_source = ColumnDataSource(dict(ts=[],
                                                   y_true=[],
                                                   y_pred=[]))
        self.pred_decoding = main_config['pred_decoding']

        # LSL stream reader
        self.lsl_reader = None
        self.lsl_start_time = None
        self.thread_lsl = QtCore.QThreadPool()
        self.channel_source = ColumnDataSource(dict(ts=[], eeg=[]))
        self._lsl_data = (None, None)

        # LSL stream recorder
        if not os.path.isdir(main_config['record_path']):
            os.mkdir(main_config['record_path'])
        self.record_path = main_config['record_path']
        self.record_name = game_config['record_name']
        self.lsl_recorder = None

        # Predictor
        self.models_path = main_config['models_path']
        self.input_signal = np.zeros((self.n_channels, 4 * self.fs))
        self.predictor = None
        self.thread_pred = QtCore.QThreadPool()
        self._pred_action = (0, 'Rest')

    @property
    def lsl_data(self):
        return self._lsl_data

    @lsl_data.setter
    def lsl_data(self, data):
        self._lsl_data = data

        # Memorize the most recent timestamp
        ts, eeg = data
        self.last_ts = ts[-1]

        # Record signal
        if self.lsl_recorder is not None:
            self.lsl_recorder.save_data(copy.deepcopy(ts), copy.deepcopy(eeg))

        self.parent.add_next_tick_callback(self.update_signal)

    @property
    def pred_action(self):
        return self._pred_action

    @pred_action.setter
    def pred_action(self, val_tuple):
        self._pred_action = val_tuple
        if self.game_start_time is not None:
            self.parent.add_next_tick_callback(self.update_prediction)

    @property
    def expected_action(self):
        return self._expected_action

    @expected_action.setter
    def expected_action(self, action):
        logging.info(f'Receiving groundtruth from logs: {action}')
        self._expected_action = copy.deepcopy(action)

        # In autoplay, we directly update the model prediction (no delay)
        if self.modelfile == 'AUTOPLAY':
            self._pred_action = copy.deepcopy(action)

        self.parent.add_next_tick_callback(self.update_groundtruth)

    @property
    def available_logs(self):
        logs = list(self.game_logs_path.glob(game_config['game_logs_pattern']))
        return sorted(logs)

    @property
    def game_is_on(self):
        if self.game is not None:
            # Poll returns None when game process is running and 0 otherwise
            return self.game.poll() is None
        else:
            return False

    @property
    def should_record(self):
        return 'Record' in self.selected_settings

    @property
    def available_models(self):
        ml_models = [p.name for p in self.models_path.glob('*.pkl')]
        dl_models = [p.name for p in self.models_path.glob('*.h5')]
        return ['AUTOPLAY'] + ml_models + dl_models

    @property
    def model_name(self):
        return self.select_model.value

    @property
    def modelfile(self):
        if self.select_model.value == 'AUTOPLAY':
            return 'AUTOPLAY'
        else:
            return self.models_path / self.select_model.value

    @property
    def is_convnet(self):
        return self.select_model.value.split('.')[-1] == 'h5'

    @property
    def available_ports(self):
        if sys.platform == 'linux':
            ports = self.micro_path.glob('*')
            return [''] + [p.name for p in ports]
        elif sys.platform == 'win32':
            return [''] + [p.device for p in serial.tools.list_ports.comports()]

    @property
    def sending_events(self):
        return 'Send events' in self.selected_settings

    @property
    def channel_idx(self):
        return int(self.select_channel.value.split('-')[0])

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def accuracy(self):
        y_pred = self.chrono_source.data['y_pred']
        y_true = self.chrono_source.data['y_true']
        return accuracy_score(y_true, y_pred)

    def reset_lsl(self):
        if self.lsl_reader:
            self.lsl_reader.should_stream = False
            self.lsl_reader = None
            self.lsl_start_time = None
            self.thread_lsl.clear()

    def reset_predictor(self):
        if self.predictor:
            self.predictor.should_predict = False
            self.predictor = None
            self.thread_pred.clear()

    def reset_recorder(self):
        if self.lsl_recorder:
            self.lsl_recorder.close_h5()
            self.lsl_recorder = None

    def reset_plots(self):
        self.chrono_source.data = dict(ts=[], y_pred=[], y_true=[])
        self.channel_source.data = dict(ts=[], eeg=[])
        self.gd_info.text = ''
        self.pred_info.text = ''
        self.acc_info.text = ''

    def reset_game(self):
        self.game.kill()
        self.game = None

    def reset_log_reader(self):
        logging.info('Delete old log reader')
        self.thread_log.clear()
        self.game_log_reader = None

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models
        self.model_info.text = f'<b>Model:</b> {new}'
        self.parent.add_next_tick_callback(self.start_predictor_thread)

    def on_select_port(self, attr, old, new):
        logging.info(f'Select new port: {new}')

        if self.port_sender is not None:
            logging.info('Delete old log reader')
            self.port_sender = None

        logging.info(f'Instanciate port sender {new}')
        self.port_sender = CommandSenderPort(new)

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data['eeg'] = []
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def on_settings_change(self, attr, old, new):
        self.plot_stream.visible = 0 in new

    def start_game_process(self):
        logging.info('Lauching Cybathlon game')
        self.n_old_logs = len(self.available_logs)
        self.reset_plots()

        # Close any previous game process
        if self.game is not None:
            self.reset_game()

        self.game = subprocess.Popen(str(self.game_path),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True)
        assert self.game is not None, 'Can\'t launch game !'

    def start_log_reader(self):
        # Check if log reader already instanciated
        if self.game_log_reader is not None:
            self.reset_log_reader()

        # Wait for new logfile to be created
        while not len(self.available_logs) - self.n_old_logs > 0:
            logging.info('Waiting for new race logs...')
            time.sleep(0.5)
        log_filename = str(self.available_logs[-1])

        # Log reader is started in a separate thread
        logging.info(f'Instanciate log reader {log_filename}')
        self.game_log_reader = GameLogReader(self, log_filename,
                                             self.player_idx)
        self.thread_log.start(self.game_log_reader)

    def on_launch_game_start(self):
        self.button_launch_game.label = 'Lauching...'
        self.button_launch_game.button_type = 'warning'
        self.parent.add_next_tick_callback(self.on_launch_game)

    def on_launch_game(self):
        self.start_game_process()
        self.start_log_reader()
        self.button_launch_game.label = 'Launched'
        self.button_launch_game.button_type = 'success'

    def update_groundtruth(self):
        action_idx, action_name = self.expected_action

        # Start autoplay predictor when game starts + reset chronogram (if multiple consecutive runs)
        if action_name == 'Game start':
            self.reset_plots()
            self.game_start_time = time.time()
            if self.modelfile == 'AUTOPLAY':
                self.parent.add_next_tick_callback(self.start_predictor_thread)
        elif action_name in ['Game end', 'Pause']:
            self.reset_predictor()
        elif action_name == 'Resume':
            self.parent.add_next_tick_callback(self.start_predictor_thread)
        elif action_name == 'Reset game':
            self.reset_plots()
            self.reset_predictor()
            self.reset_log_reader()
            self.parent.add_next_tick_callback(self.start_log_reader)

        # Send groundtruth to microcontroller
        if self.sending_events:
            if self.port_sender is not None:
                self.port_sender.sendCommand(action_idx)
                logging.info(f'Send event: {action_idx}')
            else:
                logging.info('Please select a port !')

    def update_prediction(self):
        if not self.game_is_on:
            logging.info('Game window was closed')
            self.button_launch_game.label = 'Launch Game'
            self.button_launch_game.button_type = 'primary'
            self.select_model.value = 'AUTOPLAY'
            self.reset_predictor()
            return

        groundtruth = self.expected_action[0]
        action_idx = self.pred_action[0]

        # Save groundtruth as event
        if self.lsl_recorder is not None:
            marker_id = int(f'{(groundtruth+1)*2}{(action_idx+1)*2}')
            self.lsl_recorder.save_event(self.last_ts, marker_id)

        # Update chronogram source
        ts = time.time() - self.game_start_time
        self.chrono_source.stream(dict(ts=[ts],
                                       y_true=[groundtruth],
                                       y_pred=[action_idx]))

        # Update information display
        self.gd_info.text = f'<b>Groundtruth:</b> {self.expected_action}'
        self.pred_info.text = f'<b>Prediction:</b> {self.pred_action}'
        self.acc_info.text = f'<b>Accuracy:</b> {self.accuracy:.2f}'

    def on_lsl_connect_toggle(self, active):
        if active:
            # Connect to LSL stream
            self.button_lsl.label = 'Seaching...'
            self.button_lsl.button_type = 'warning'
            self.parent.add_next_tick_callback(self.start_lsl_thread)
        else:
            self.reset_lsl()
            self.button_lsl.label = 'LSL Disconnected'
            self.button_lsl.button_type = 'danger'

    def start_lsl_thread(self):
        try:
            self.lsl_reader = LSLClient(self)
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

    def start_predictor_thread(self):
        self.reset_predictor()

        try:
            self.predictor = ActionPredictor(self,
                                             self.modelfile,
                                             self.is_convnet)
            self.thread_pred.start(self.predictor)
        except Exception as e:
            logging.error(f'Failed loading model {self.modelfile} - {e}')
            self.select_model.value = 'AUTOPLAY'
            self.reset_predictor()

    def on_lsl_record_toggle(self, active):
        if active:
            try:
                self.lsl_recorder = LSLRecorder(self.record_path,
                                                self.record_name,
                                                self.lsl_reader.ch_names)
                self.lsl_recorder.open_h5()
                self.button_record.label = 'Stop recording'
                self.button_record.button_type = 'success'
            except Exception:
                self.reset_recorder()
                self.button_record.label = 'Recording failed'
                self.button_record.button_type = 'danger'
        else:
            self.reset_recorder()
            self.button_record.label = 'Start recording'
            self.button_record.button_type = 'primary'

    def update_signal(self):
        ts, eeg = self.lsl_data

        if ts.shape[0] != eeg.shape[-1]:
            logging.info('Skipping data points (bad format)')
            return

        # Convert timestamps in seconds
        if self.lsl_start_time is None:
            self.lsl_start_time = time.time()
            self.t0 = ts[0]

        # Update source display
        ch = self.channel_idx
        self.channel_source.stream(dict(ts=ts-self.t0, eeg=eeg[ch, :]),
                                   rollover=int(2 * self.fs))

        # Update signal
        chunk_size = eeg.shape[-1]
        self.input_signal = np.roll(self.input_signal, -chunk_size, axis=-1)
        self.input_signal[:, -chunk_size:] = eeg

    def create_widget(self):
        # Button - Launch Cybathlon game in new window
        self.button_launch_game = Button(label='Launch Game',
                                         button_type='primary')
        self.button_launch_game.on_click(self.on_launch_game_start)

        # Toggle - Connect to LSL stream
        self.button_lsl = Toggle(label='Connect to LSL')
        self.button_lsl.on_click(self.on_lsl_connect_toggle)

        # Toggle - Start/stop LSL stream recording
        self.button_record = Toggle(label='Start Recording',
                                    button_type='primary')
        self.button_record.on_click(self.on_lsl_record_toggle)

        # Select - Choose pre-trained model
        self.select_model = Select(title="Select pre-trained model",
                                   value='AUTOPLAY',
                                   options=self.available_models)
        self.select_model.on_change('value', self.on_model_change)

        # Select - Choose port to send events to
        self.select_port = Select(title='Select port')
        self.select_port.options = self.available_ports
        self.select_port.on_change('value', self.on_select_port)

        # Checkbox - Choose player settings
        self.div_settings = Div(text='<b>Settings</b>', align='center')
        self.checkbox_settings = CheckboxButtonGroup(labels=['Show signal',
                                                             'Send events'])
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
        self.plot_chronogram.line(x='ts', y='y_true', color='blue',
                                  source=self.chrono_source,
                                  legend_label='Groundtruth')
        self.plot_chronogram.cross(x='ts', y='y_pred', color='red',
                                   source=self.chrono_source,
                                   legend_label='Prediction')
        self.plot_chronogram.legend.background_fill_alpha = 0.6
        self.plot_chronogram.yaxis.ticker = list(self.pred_decoding.keys())
        self.plot_chronogram.yaxis.major_label_overrides = self.pred_decoding

        # Div - Display useful information
        self.model_info = Div(text=f'<b>Model:</b> AUTOPLAY')
        self.pred_info = Div()
        self.gd_info = Div()
        self.acc_info = Div()

        # Create layout
        column1 = column(self.button_launch_game, self.button_lsl,
                         self.button_record, self.select_model,
                         self.select_port, self.select_channel,
                         self.div_settings, self.checkbox_settings)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.model_info, self.gd_info,
                         self.pred_info, self.acc_info)
        return row(column1, column2, column3)
