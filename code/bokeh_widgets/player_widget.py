import glob
import logging
import socket
import subprocess
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Button, Div, Select
from bokeh.models import CheckboxButtonGroup
from pylsl import StreamInlet, resolve_streams
from pyqtgraph.Qt import QtCore

from feature_extraction_functions.models import load_model
from preprocessing_functions.preproc_functions import filtering, rereference
from preprocessing_functions.preproc_functions import clipping, standardize


class PlayerWidget:
    def __init__(self, parent=None):
        self.player_idx = 1

        # Game log reader (separate thread)
        self.game_log_reader = None
        self.game_logs_path = '../game/log/raceLog*.txt'
        self.ports = '/dev/ttyACM*'
        self._expected_action = (3, "Rest")
        self.thread_log = QtCore.QThreadPool()

        # Port event sender
        self.port_sender = None

        # Game player
        self.game_player = GamePlayer(self.player_idx)

        # LSL stream reader
        self.parent = parent
        self.lsl_reader = None
        self.signal_source = ColumnDataSource(dict(ts=[], data=[]))
        self.thread_lsl = QtCore.QThreadPool()
        self.callback_lsl_id = None

        # Model
        self.model = None
        self.signal = None
        self._last_pred = None
        self.callback_pred_id = None

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def selected_preproc(self):
        active = self.checkbox_preproc.active
        return [self.checkbox_preproc.labels[i] for i in active]

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
        action_idx, action_name = action
        self._expected_action = action

        if self.autoplay:
            logging.info(f'Autoplay: {action}')
            self.game_player.sendCommand(action_idx)

        if self.sending_events:
            if self.port_sender is not None:
                self.port_sender.sendCommand(action_idx)
                logging.info(f'Send event: {action}')
            else:
                logging.info('Could not send event')

    @property
    def get_ports(self):
        if sys.platform == 'linux':
            return glob.glob(self.ports)
        elif sys.platform == 'win32':
            return [p.device for p in serial.tools.list_ports.comports()]

    @property
    def model_name(self):
        return self.select_model.value

    def on_launch_game(self):
        logging.info('Lauching Cybathlon game')
        game = subprocess.Popen('../game/brainDriver', shell=False)
        assert game is not None, 'Can\'t launch game !'

        # Wait for logfile to be created
        time.sleep(5)
        log_filename = glob.glob(self.game_logs_path)[-1]

        # Check if log reader already instanciated
        if self.game_log_reader is not None:
            logging.info('Delete old log reader')
            self.thread_log.clear()
            del self.game_log_reader

        logging.info('Instanciate log reader')
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
        self.select_port.options = [''] + self.get_ports
        self.select_model.options = [''] + glob.glob('./saved_models/*.pkl')

        if self.sending_events:
            assert self.port_sender is not None, 'Select port first !'
            logging.info('Active events sending')

        if self.autoplay and not self.should_predict:
            assert self.game_log_reader is not None, 'Select log filename first !'
            logging.info('Activate autoplay')
            self.game_player.sendCommand(self.expected_action[0])

        elif self.should_predict and not self.autoplay:
            assert self.model is not None, 'Load pre-trained model first !'
            logging.info('Activate model prediction')
            self.callback_pred_id = self.parent.add_periodic_callback(self.callback_pred,
                                                                      250)

    def callback_pred(self):
        X = np.copy(self.signal)
        X = np.delete(X, [0, 30], axis=0)
        logging.info(f'{X.shape} - {self.signal.shape}')

        # Preprocessing
        if 'Filter' in self.selected_preproc:
            X = filtering(X, f_low=0.5, f_high=38,
                          fs=500, f_order=3)

        if 'Standardize' in self.selected_preproc:
            X = clipping(X, 6)
            X = standardize(X)

        if 'Rereference' in self.selected_preproc:
            X = rereference(X)

        # Predict on the most recent 3.5s of signal
        action_idx = self.model.predict(X[np.newaxis, :, 100:1850])[0]
        assert action_idx in [0, 1, 2, 3], \
            'Prediction is not in allowed action space'

        # Send command
        self.game_player.sendCommand(action_idx)
        self._last_pred = action_idx

        self.div_info.text = f'<b>Model:</b> {self.model_name} <br>' \
            f'<b>Groundtruth:</b> {self._expected_action} <br>' \
            f'<b>Prediction:</b> {self._last_pred} <br>'

    def on_model_change(self, attr, old, new):
        logging.info(f'Select new pre-trained model {new}')
        self.model = load_model(new)
        self.select_model.options = [''] + glob.glob('./saved_models/*.pkl')
        # TODO: Deep learning case

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
                self.select_channel.options = [f'{i} - {ch}' for i, ch
                                               in enumerate(self.lsl_reader.ch_names)]
                self.callback_lsl_id = self.parent.add_periodic_callback(self.callback_lsl,
                                                                         100)
                self.button_lsl.button_type = 'success'
        except Exception as e:
            logging.info(e)
            self.lsl_reader = None

    def callback_lsl(self):
        ''' Fetch EEG data from LSL stream '''
        data, ts = [], []
        try:
            data, ts = self.lsl_reader.get_data()

            if len(data.shape) > 1:
                # Clean signal and reference TODO: just for visualization purpose
                data = np.swapaxes(data, 1, 0)
                data = 1e6 * data

                # Update source
                self.signal_source.stream(dict(ts=ts,
                                               data=data[0, :]),
                                          rollover=1000)
                # Update signal
                chunk_size = data.shape[-1]
                self.signal = np.roll(self.signal, -chunk_size, axis=-1)
                self.signal[:, -chunk_size:] = data

        except Exception as e:
            logging.info(f'Ending periodic callback - {e}')
            self.button_lsl.button_type = 'warning'
            if self.callback_lsl_id:
                self.parent.remove_periodic_callback(self.callback_lsl_id)
            if self.callback_pred_id:
                self.parent.remove_periodic_callback(self.callback_pred_id)

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
        self.select_model.options = [''] + glob.glob('./saved_models/*.pkl')
        self.select_model.on_change('value', self.on_model_change)

        # Select - Choose port to send events to
        self.select_port = Select(title='Select port', options=[''])
        self.select_port.options += self.get_ports
        self.select_port.on_change('value', self.on_select_port)

        # Checkbox - Choose player settings
        self.div_settings = Div(text='<b>Settings</b>', align='center')
        self.checkbox_settings = CheckboxButtonGroup(labels=['Autoplay',
                                                             'Predict',
                                                             'Send events'])
        self.checkbox_settings.on_click(self.on_checkbox_settings_change)

        # Checkbox - Choose preprocessing steps
        self.div_preproc = Div(text='<b>Preprocessing</b>', align='center')
        self.checkbox_preproc = CheckboxButtonGroup(labels=['Filter',
                                                            'Standardize',
                                                            'Rereference'])

        # Select - Channel to visualize TODO: get channel names for LSL to get
        self.select_channel = Select(title='Select channel')

        # Plot - LSL EEG Stream
        self.plot_stream = figure(title='Temporal EEG signal',
                                  x_axis_label='Time [s]',
                                  y_axis_label='Amplitude',
                                  plot_height=500,
                                  plot_width=800)
        self.plot_stream.line(x='ts', y='data', source=self.signal_source)

        # Div - Display useful information
        self.div_info = Div()

        # Create layout
        column1 = column(self.button_launch_game, self.button_lsl,
                         self.select_model, self.select_port,
                         self.select_channel,
                         self.div_settings, self.checkbox_settings,
                         self.div_preproc, self.checkbox_preproc)
        column2 = column(self.plot_stream)
        column3 = column(self.div_info)
        return row(column1, column2, column3)


class GameLogReader(QtCore.QRunnable):
    def __init__(self, parent, logfilename, player_idx):
        super(GameLogReader, self).__init__()
        self.parent = parent
        self.logfilename = logfilename
        self.player_idx = player_idx
        self.log2actions = {"leftWinker": (0, "Left"),
                            "rightWinker": (1, "Right"),
                            "headlight": (2, "Light"),
                            "none": (3, "Rest")}

    def notify(self, expected_action):
        self.parent.expected_action = expected_action

    def follow(self, thefile):
        thefile.seek(0, 2)
        while True:
            line = thefile.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

    @QtCore.pyqtSlot()
    def run(self):
        logfile = open(self.logfilename, "r")
        loglines = self.follow(logfile)

        for line in loglines:
            if ("p{}_expectedInput".format(self.player_idx) in line):
                if "none" in line:
                    expected_action = self.log2actions['none']
                else:
                    action = line.split(" ")[-1].strip()
                    expected_action = self.log2actions[action]

                logging.info(f"Groundtruth: {expected_action[1]}")
                self.notify(expected_action)


class GamePlayer:
    def __init__(self, player_idx):
        # Send commands using a separate thread
        self.thread_game = QtCore.QThreadPool()

    def sendCommand(self, action_idx):
        ''' Send the command to the game after a delay in a separate thread '''
        if action_idx not in [None, 3]:
            command_sender = CommandSenderGame(action_idx)
            self.thread_game.start(command_sender)


class CommandSenderGame(QtCore.QRunnable):
    def __init__(self, action_idx):
        super(CommandSenderGame, self).__init__()
        self.action_idx = action_idx

        # Communication protocol with game
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5555
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.commands = {0: "\x0B", 1: "\x0D", 2: "\x0C", 3: ""}

    @QtCore.pyqtSlot()
    def run(self):
        # TODO: only random delay when autoplay
        time.sleep(np.random.random_sample())
        self.sock.sendto(bytes(self.commands[self.action_idx], "utf-8"),
                         (self.UDP_IP, self.UDP_PORT))


class CommandSenderPort:
    def __init__(self, port='/dev/ttyACM0'):
        logging.info(f'Port: {port}')
        self.serial = serial.Serial(port=port,
                                    baudrate=115200,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    bytesize=serial.EIGHTBITS)

    def sendCommand(self, action_idx):
        message = str(action_idx)
        self.serial.write(message.encode())


class LSLClient(QtCore.QRunnable):
    def __init__(self):
        super().__init__()

        logging.info('Looking for LSL stream...')
        available_streams = resolve_streams(5)

        if len(available_streams) > 0:
            self.stream_reader = StreamInlet(available_streams[0],
                                             max_chunklen=1,
                                             recover=False)
            id = self.stream_reader.info().session_id()
            self.fs = int(self.stream_reader.info().nominal_srate())
            self.n_channels = int(self.stream_reader.info().channel_count())
            logging.info(
                f'Stream {id} found at {self.fs} Hz with {self.n_channels} channels')

            # Fetching channel names
            ch = self.stream_reader.info().desc().child('channels').first_child()
            self.ch_names = []
            for i in range(self.n_channels):
                self.ch_names.append(ch.child_value('label'))
                ch = ch.next_sibling()
            logging.info(f"Channel names: {self.ch_names}")
        else:
            logging.error('No stream found !')
            raise Exception

    def get_data(self):
        try:
            # Data is of shape (n_timestamps, n_channels)
            data, ts = self.stream_reader.pull_chunk()
        except Exception as e:
            logging.info(f'{e} - No more data')
        return np.array(data), np.array(ts)
