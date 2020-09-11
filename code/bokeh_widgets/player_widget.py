import logging
import subprocess
import sys
import time
from pathlib import Path
import traceback

import numpy as np
import serial
import serial.tools.list_ports
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models import Button, Div, Select, CheckboxButtonGroup, RadioButtonGroup
from pyqtgraph.Qt import QtCore
from sklearn.metrics import accuracy_score

from src.pipeline import load_pipeline
from src.models import predict
from src.lsl_client import LSLClient
from src.game_player import GamePlayer, CommandSenderPort
from src.game_log_reader import GameLogReader
from .utils import clean_log_directory


class PlayerWidget:
    def __init__(self, parent=None):
        self.player_idx = 1
        self.fs = 500
        self.n_channels = 61

        # Game log reader (separate thread)
        self.game_logs_path = Path('../game/log')
        self.game_log_reader = None
        self.ports = Path('/dev/ttyACM*')
        self._expected_action = None
        self.thread_log = QtCore.QThreadPool()

        # Game window (separate process)
        clean_log_directory(self.game_logs_path)
        self.game = None

        # Port event sender
        self.port_sender = None

        # Game player
        self.game_path = Path('../game/brainDriver')
        self.game_player = GamePlayer(self.player_idx)
        self.game_start_time = None
        self.callback_action_id = None
        self.chrono_source = ColumnDataSource(dict(ts=[], y_true=[],
                                                   y_pred=[]))
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
        self.n_crops = 10
        self.crop_len = 0.5

    @property
    def available_models(self):
        ml_models = [p.name for p in self.models_path.glob('*.pkl')]
        dl_models = [p.name for p in self.models_path.glob('*.h5')]
        return [''] + ml_models + dl_models

    @property
    def available_ports(self):
        if sys.platform == 'linux':
            ports = self.ports.glob('*')
            return [''] + [p.name for p in ports]
        elif sys.platform == 'win32':
            return [''] + [p.device for p in serial.tools.list_ports.comports()]

    @property
    def available_logs(self):
        logs = list(self.game_logs_path.glob('raceLog*.txt'))
        return sorted(logs)

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def model_path(self):
        return self.models_path / self.select_model.value

    @property
    def autoplay(self):
        return self.radio_mode.active == 0

    @property
    def should_predict(self):
        return self.radio_mode.active == 1

    @property
    def sending_events(self):
        return 'Send events' in self.selected_settings

    @property
    def expected_action(self):
        return self._expected_action

    @expected_action.setter
    def expected_action(self, action):
        logging.info(f'Receiving expected action {action}')
        action_idx, action_name = action
        self._expected_action = action

        # Reset timer & start action callback (autopilot/prediction)
        if action_name == 'Game start':
            logging.info('Game start')
            self.parent.add_next_tick_callback(self.clear_chronogram)
            self.game_start_time = time.time()
            self.parent.add_next_tick_callback(self.create_action_callback)
        elif action_name == 'Game end':
            logging.info('Game end')
            self.parent.add_next_tick_callback(self.remove_action_callback)
        elif action_name == 'Pause':
            logging.info('Pause game')
            self.parent.add_next_tick_callback(self.remove_action_callback)
        elif action_name == 'Resume':
            logging.info('Resume game')
            self.parent.add_next_tick_callback(self.create_action_callback)

        # Send groundtruth to microcontroller
        if self.sending_events:
            if self.port_sender is not None:
                self.port_sender.sendCommand(action_idx)
                logging.info(f'Send event: {action}')
            else:
                logging.info('Please select a port !')

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

    @property
    def accuracy(self):
        y_pred = self.chrono_source.data['y_pred']
        y_true = self.chrono_source.data['y_true']
        return accuracy_score(y_true, y_pred)

    @property
    def game_is_on(self):
        if self.game is not None:
            # Poll returns None when game process is running and 0 otherwise
            return self.game.poll() is None
        else:
            return False

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models
        self.model = load_pipeline(self.model_path)

    def on_select_port(self, attr, old, new):
        logging.info(f'Select new port: {new}')

        if self.port_sender is not None:
            logging.info('Delete old log reader')
            del self.port_sender

        logging.info(f'Instanciate port sender {new}')
        self.port_sender = CommandSenderPort(new)

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data['data'] = []
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def on_mode_change(self, active):
        logging.info(f'Mode: {self.radio_mode.labels[active]}')

    def reset_all(self):
        self.reset_game()
        self.reset_log_reader()
        self.remove_action_callback()
        self.remove_lsl_callback()

    def reset_game(self):
        self.game.kill()
        self.game = None

    def reset_log_reader(self):
        logging.info('Delete old log reader')
        self.thread_log.clear()
        del self.game_log_reader

    def clear_chronogram(self):
        logging.info('Clear chronogram')
        self.chrono_source.data = dict(ts=[], y_true=[],
                                       y_pred=[])
        self.div_info.text = ''

    def start_game_process(self):
        logging.info('Lauching Cybathlon game')
        self.n_old_logs = len(self.available_logs)

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
        self.radio_mode.active = 0
        self.button_launch_game.label = 'Lauching...'
        self.button_launch_game.button_type = 'warning'
        self.parent.add_next_tick_callback(self.on_launch_game)

    def on_launch_game(self):
        self.start_game_process()
        self.start_log_reader()
        self.button_launch_game.label = 'Launched'
        self.button_launch_game.button_type = 'success'

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

        if not self.game_is_on:
            logging.info('Game window was closed')
            self.button_launch_game.label = 'Launch Game'
            self.button_launch_game.button_type = 'primary'
            self.parent.add_next_tick_callback(self.remove_action_callback)
            return

        # Case 1: Autopilot - Return expected action from logs
        if self.autoplay:
            model_name = 'Autoplay'
            # time.sleep(np.random.random_sample())
            action_idx = self.expected_action[0]
            print(f'Game state: {self.game.poll()}')
            time.sleep(0.5)

        # Case 2: Model prediction - Predict from LSL stream TODO: extract this as a function
        elif self.should_predict:
            assert self.callback_lsl_id is not None, 'Please connect to LSL stream'
            model_name = self.model_name

            X = np.copy(self.signal)

            # Removing FP1 & FP2 TODO: Don't hardcode
            X = np.delete(X, [0, 30], axis=0)

            # Selecting last 1s of signal
            X = X[np.newaxis, :, -self.fs:]

            # If no model
            if self.model is None:
                action_idx = 0
                logging.warning('Rest action sent by default!'
                                'Please select a model.')
            else:
                action_idx = predict(X, self.model, self.is_convnet,
                                     self.n_crops, self.crop_len, self.fs,
                                     self.should_reref, self.should_filter,
                                     self.should_standardize)
                logging.info(f'Action idx: {action_idx}')

        self.current_pred = (action_idx, self.pred2encoding[action_idx])

        if self.game_start_time is not None and self.game_is_on:
            # Send action to game avatar (if not rest command)
            if action_idx in [1, 2, 3]:
                logging.info(f'Sending: {action_idx}')
                self.game_player.sendCommand(action_idx)

            # Update chronogram source (if race started)
            ts = time.time() - self.game_start_time
            self.chrono_source.stream(dict(ts=[ts],
                                           y_true=[self.expected_action[0]],
                                           y_pred=[action_idx]))

        # Update information display
        self.div_info.text = f'<b>Model:</b> {model_name} <br>' \
            f'<b>Groundtruth:</b> {self.expected_action} <br>' \
            f'<b>Prediction:</b> {self.current_pred} <br>' \
            f'<b>Accuracy:</b> {self.accuracy:.2f} <br>'

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
                self.radio_mode.active = 1

        except Exception:
            logging.info(f'No LSL stream - {traceback.format_exc()}')
            self.button_lsl.label = 'Can\'t find stream'
            self.button_lsl.button_type = 'danger'
            self.lsl_reader = None

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
            self.lsl_start_time = ts[0]
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
        # Button - Launch Cybathlon game in new window
        self.button_launch_game = Button(label='Launch Game',
                                         button_type='primary')
        self.button_launch_game.on_click(self.on_launch_game_start)

        # Button - Connect to LSL stream
        self.button_lsl = Button(label='Connect to LSL')
        self.button_lsl.on_click(self.on_lsl_connect_start)

        # Select - Choose pre-trained model
        self.select_model = Select(title="Select pre-trained model")
        self.select_model.options = self.available_models
        self.select_model.on_change('value', self.on_model_change)

        # Select - Choose port to send events to
        self.select_port = Select(title='Select port')
        self.select_port.options = self.available_ports
        self.select_port.on_change('value', self.on_select_port)

        # Radio button - Choose play mode
        self.div_mode = Div(text='<b>Mode</b>', align='center')
        self.radio_mode = RadioButtonGroup(labels=['Autoplay', 'Model preds'])
        self.radio_mode.on_click(self.on_mode_change)

        # Checkbox - Choose player settings
        self.div_settings = Div(text='<b>Settings</b>', align='center')
        self.checkbox_settings = CheckboxButtonGroup(labels=['Send events'])

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
                                  plot_width=800)
        self.plot_stream.line(x='ts', y='data', source=self.channel_source)

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
        self.plot_chronogram.legend.background_fill_alpha = 0.8
        self.plot_chronogram.yaxis.ticker = list(self.pred2encoding.keys())
        self.plot_chronogram.yaxis.major_label_overrides = self.pred2encoding

        # Div - Display useful information
        self.div_info = Div()

        # Create layout
        column1 = column(self.button_launch_game, self.button_lsl,
                         self.select_model, self.select_port,
                         self.select_channel,
                         self.div_mode, self.radio_mode,
                         self.div_settings, self.checkbox_settings,
                         self.div_preproc, self.checkbox_preproc)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.div_info)
        return row(column1, column2, column3)
