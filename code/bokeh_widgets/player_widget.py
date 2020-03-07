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
from bokeh.models.widgets import Button, Div, Select, CheckboxButtonGroup
from pyqtgraph.Qt import QtCore
from sklearn.metrics import accuracy_score

from src.dataloader import preprocessing
from src.models import load_model
from src.lsl_client import LSLClient
from src.game_player import GamePlayer, CommandSenderPort
from src.game_log_reader import GameLogReader


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
        self.thread_lsl = QtCore.QThreadPool()
        self.callback_lsl_id = None
        self.channel_source = ColumnDataSource(dict(ts=[], data=[]))

        # Model
        self.models_path = Path('./saved_models')
        self.model = None
        self.signal = None
        self._last_pred = (3, "Rest")

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
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def model_path(self):
        return self.models_path / self.select_model.value

    @property
    def autoplay(self):
        return 'Autoplay' in self.selected_settings

    @property
    def should_predict(self):
        return 'Predict' in self.selected_settings

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
            self.game_start_time = time.time()
            self.parent.add_next_tick_callback(self.create_action_callback)

        # Send groundtruth to microcontroller
        if self.sending_events:
            assert self.port_sender is not None, 'Please select a port !'
            self.port_sender.sendCommand(action_idx)
            logging.info(f'Send event: {action}')

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
    def should_standardize(self):
        return 'Standardize' in self.selected_preproc

    @property
    def should_crop(self):
        return 'Crop' in self.selected_preproc

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

    def on_launch_game(self):
        logging.info('Lauching Cybathlon game')
        game = subprocess.Popen(str(self.game_path), shell=False)
        assert game is not None, 'Can\'t launch game !'

        # Wait for logfile to be created
        time.sleep(5)
        logs = list(self.game_logs_path.glob('raceLog*.txt'))
        log_filename = str(logs[-1])

        # Check if log reader already instanciated
        if self.game_log_reader is not None:
            logging.info('Delete old log reader')
            self.thread_log.clear()
            del self.game_log_reader

        logging.info(f'Instanciate log reader {log_filename}')
        self.game_log_reader = GameLogReader(self, log_filename,
                                             self.player_idx)
        self.thread_log.start(self.game_log_reader)
        self.button_launch_game.button_type = 'success'

    def on_select_port(self, attr, old, new):
        logging.info(f'Select new port: {new}')

        if self.port_sender is not None:
            logging.info('Delete old log reader')
            del self.port_sender

        logging.info(f'Instanciate port sender {new}')
        self.port_sender = CommandSenderPort(new)

    def on_checkbox_settings_change(self, active):
        logging.info(f'Active settings: {active}')
        self.select_port.options = self.available_ports
        self.select_model.options = self.available_models

        if self.autoplay and self.should_predict:
            logging.info('Deactivate autoplay first !')
            self.checkbox_settings.active = [0]

        elif self.should_predict and self.callback_action_id is None:
            if self.model is not None:
                self.create_action_callback()
            else:
                logging.info('Load pre-trained model first !')
                self.checkbox_settings.active = [0]

        elif self.autoplay and self.callback_action_id is None:
            self.create_action_callback()

        if self.sending_events and self.port_sender is None:
            logging.info('Select port first !')
            self.checkbox_settings.active = [0]

    def predict(self):
        assert self.callback_lsl_id is not None, 'Please connect to LSL stream'
        X = np.copy(self.signal)

        # Removing FP1 & FP2 TODO: Don't hardcode
        X = np.delete(X, [0, 30], axis=0)

        # Selecting last 1s of signal
        X = X[:, :, -self.fs:]

        # Preprocessing
        X = preprocessing(X,
                          rereference=self.should_reref,
                          standardize=self.should_standardize,
                          crop=self.should_crop,
                          dl_shape=self.is_convnet)

        # TODO: Average 10 predictions using 1s of signal
        action_idx = self.model.predict(X)[0]
        assert action_idx in [0, 1, 2, 3], \
            'Prediction is not in allowed action space'
        return action_idx

    def create_action_callback(self):
        assert self.callback_action_id is None, 'Action callback already exists!'
        logging.info('Create action callback')
        self.callback_action_id = self.parent.add_periodic_callback(
            self.callback_action, 1000)

    def remove_action_callback(self):
        assert self.callback_action_id is not None, 'Action callback doesn\'t exist'
        logging.info('Remove action callback')
        self.parent.remove_periodic_callback(self.callback_action_id)
        self.callback_action_id = None

    def callback_action(self):
        ''' This periodic callback starts at the same time that the race '''
        # Case 1: Autopilot - Return expected action from logs
        if self.autoplay:
            model_name = 'Autoplay'
            # time.sleep(np.random.random_sample())
            action_idx = self.expected_action[0]

        # Case 2: Model prediction - Predict from LSL stream
        elif self.should_predict:
            model_name = self.model_name
            action_idx = self.predict()

        # Case 3: Remove callback
        else:
            self.remove_action_callback()
            return

        # Send action to game avatar (if not rest command)
        if action_idx in [0, 1, 2]:
            logging.info(f'Sending: {action_idx}')
            self.game_player.sendCommand(action_idx)

        self._last_pred = (action_idx, self.pred2encoding[action_idx])

        # Update chronogram source (if race started)
        if self.game_start_time is not None:
            ts = time.time() - self.game_start_time
            self.chrono_source.stream(dict(ts=[ts],
                                           y_true=[self.expected_action[0]],
                                           y_pred=[action_idx]))

        # Update information display
        self.div_info.text = f'<b>Model:</b> {model_name} <br>' \
            f'<b>Groundtruth:</b> {self.expected_action} <br>' \
            f'<b>Prediction:</b> {self._last_pred} <br>' \
            f'<b>Accuracy:</b> {self.accuracy:.2f} <br>'

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.select_model.options = self.available_models
        self.model = load_model(self.model_path)

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select new channel {new}')
        self.channel_source.data['data'] = []
        self.plot_stream.yaxis.axis_label = f'Amplitude ({new})'

    def on_lsl_connect(self):
        if self.lsl_reader is not None:
            logging.info('Delete old lsl stream')
            self.thread_lsl.clear()
            del self.lsl_reader

        try:
            self.lsl_reader = LSLClient()
            self.signal = np.zeros((self.lsl_reader.n_channels, 2000))
            if self.lsl_reader is not None:
                logging.info('Start periodic callback - LSL')
                self.select_channel.options = [f'{i+1} - {ch}' for i, ch
                                               in enumerate(self.lsl_reader.ch_names)]
                self.create_lsl_callback()
                self.button_lsl.label = 'Reading LSL stream'
                self.button_lsl.button_type = 'success'
        except Exception:
            logging.info(f'No LSL stream - {traceback.format_exc()}')
            self.lsl_reader = None

    def create_lsl_callback(self):
        assert self.callback_lsl_id is None, 'LSL callback already exists!'
        self.callback_lsl_id = self.parent.add_periodic_callback(
            self.callback_lsl, 100)

    def remove_lsl_callback(self):
        assert self.callback_lsl_id is not None, 'LSL callback doesn\'t exist'
        self.parent.remove_periodic_callback(self.callback_lsl_id)
        self.callback_lsl_id = None

    def callback_lsl(self):
        ''' Fetch EEG data from LSL stream '''
        data, ts = [], []
        try:
            data, ts = self.lsl_reader.get_data()

            # Convert timestamps in seconds
            ts /= self.fs

            if len(data.shape) > 1:
                # Clean signal and reference TODO: just for visualization purpose
                data = np.swapaxes(data, 1, 0)
                # data = 1e6 * data

                # Update source
                ch = self.channel_idx
                self.channel_source.stream(dict(ts=ts,
                                                data=data[ch, :]),
                                           rollover=int(2*self.fs))
                # Update signal
                chunk_size = data.shape[-1]
                self.signal = np.roll(self.signal, -chunk_size, axis=-1)
                self.signal[:, -chunk_size:] = data

        except Exception:
            logging.info(
                f'Ending periodic callback - {traceback.format_exc()}')
            self.remove_lsl_callback()
            self.remove_action_callback()
            self.button_lsl.label = 'No LSL stream'
            self.button_lsl.button_type = 'warning'

    def create_widget(self):
        # Button - Launch Cybathlon game in new window
        self.button_launch_game = Button(label='Launch Game',
                                         button_type='primary')
        self.button_launch_game.on_click(self.on_launch_game)

        # Button - Connect to LSL stream
        self.button_lsl = Button(label='Connect to LSL')
        self.button_lsl.on_click(self.on_lsl_connect)

        # Select - Choose pre-trained model
        self.select_model = Select(title="Select pre-trained model")
        self.select_model.options = self.available_models
        self.select_model.on_change('value', self.on_model_change)

        # Select - Choose port to send events to
        self.select_port = Select(title='Select port')
        self.select_port.options = self.available_ports
        self.select_port.on_change('value', self.on_select_port)

        # Checkbox - Choose player settings
        self.div_settings = Div(text='<b>Settings</b>', align='center')
        self.checkbox_settings = CheckboxButtonGroup(labels=['Autoplay',
                                                             'Predict',
                                                             'Send events'],
                                                     active=[0])
        self.checkbox_settings.on_click(self.on_checkbox_settings_change)

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference',
                                                            'Crop'])

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
                         self.div_settings, self.checkbox_settings,
                         self.div_preproc, self.checkbox_preproc)
        column2 = column(self.plot_stream, self.plot_chronogram)
        column3 = column(self.div_info)
        return row(column1, column2, column3)
