import numpy as np
import sys
import logging
import glob
import socket
import subprocess
import time
import serial
import serial.tools.list_ports
from pyqtgraph.Qt import QtCore
from bokeh.models.widgets import Div, Select, Button, Toggle
from bokeh.layouts import widgetbox


class PlayerWidget:
    def __init__(self):
        self.player_idx = 1

        # Game log reader (separate thread)
        self.game_log_reader = None
        self.game_logs_path = '../game/log/raceLog*.txt'
        self.ports = '/dev/ttyACM*'
        self.log_filename = ''
        self._expected_action = (3, "Rest")
        self.threadpool = QtCore.QThreadPool()

        # Port event sender
        self.port_sender = None

        # Game player
        self.game_player = GamePlayer(self.player_idx)

    @property
    def autoplay(self):
        return self.toggle_autoplay.active

    @property
    def sending_events(self):
        return self.toggle_send_events.active

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
    def selected_port(self):
        return self.select_port.value

    @property
    def get_ports(self):
        if sys.platform == 'linux':
            return glob.glob(self.ports)
        elif sys.platform == 'win32':
            return [p.device for p in serial.tools.list_ports.comports()]

    def on_launch_game(self):
        logging.info('Lauching Cybathlon game')
        game = subprocess.Popen('../game/brainDriver', shell=False)
        assert game is not None, 'Can\'t launch game !'
        time.sleep(5)
        self.select_logfile.options = [''] + glob.glob(self.game_logs_path)

    def on_select_logfile(self, attr, old, new):
        logging.info(f'Select new log file: {new}')
        self.log_filename = new

        if self.game_log_reader is None:
            logging.info('Instanciate log reader')
            self.game_log_reader = GameLogReader(self, new, self.player_idx)
            self.threadpool.start(self.game_log_reader)
        else:
            logging.info('Delete old log reader')
            self.threadpool.clear()
            del self.game_log_reader
            logging.info('Instanciate new log reader')
            self.game_log_reader = GameLogReader(self, new, self.player_idx)
            self.threadpool.start(self.game_log_reader)

    def on_select_port(self, attr, old, new):
        logging.info(f'Select new port: {new}')
        if self.port_sender is None:
            logging.info('Instanciate port sender')
            logging.info(f'Ports: {self.selected_port}')
            self.port_sender = CommandSenderPort(new)
        else:
            logging.info('Delete old log reader')
            del self.port_sender
            logging.info('Instanciate port sender')

    def on_toggle_autoplay(self, active):
        assert self.game_log_reader is not None, 'Select log filename first !'

        if active:
            logging.info('Active autoplay')
            self.toggle_autoplay.button_type = 'success'
            self.game_player.sendCommand(self.expected_action[0])

        else:
            logging.info('Inactive autoplay')
            self.toggle_autoplay.button_type = 'warning'

    def on_toggle_send_events(self, active):
        # Update list of port devices
        self.select_port.options = [''] + self.get_ports

        assert self.port_sender is not None, 'Select port first !'
        if active:
            logging.info('Active events sending')
            self.toggle_send_events.button_type = 'success'
        else:
            logging.info('Inactive events sending')
            self.toggle_send_events.button_type = 'warning'

    def create_widget(self):
        self.widget_title = Div(text='<b>Player</b>',
                                align='center', style={'color': '#000000'})

        # Button - Launch Cybathlon game in new window
        self.button_launch_game = Button(label='Launch Game',
                                         button_type='primary')
        self.button_launch_game.on_click(self.on_launch_game)

        # Select - Choose game log file to read expected events
        self.select_logfile = Select(title='Select game log file',
                                     options=[''])
        self.select_logfile.options += glob.glob(self.game_logs_path)
        self.select_logfile.on_change('value', self.on_select_logfile)

        # Toggle - Game is playing in autopilot using logs
        self.toggle_autoplay = Toggle(label='Autoplay',
                                      button_type="warning")
        self.toggle_autoplay.on_click(self.on_toggle_autoplay)

        # Select - Choose port to send events to
        self.select_port = Select(title='Select port', options=[''])
        self.select_port.options += self.get_ports
        self.select_port.on_change('value', self.on_select_port)

        # Toggle - Send game events to microcontroller port
        self.toggle_send_events = Toggle(label='Send events',
                                         button_type="warning")
        self.toggle_send_events.on_click(self.on_toggle_send_events)

        # Create layout
        layout = widgetbox([self.widget_title, self.button_launch_game,
                            self.select_logfile, self.toggle_autoplay,
                            self.select_port, self.toggle_send_events])
        return layout


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
        self.threadpool = QtCore.QThreadPool()

    def sendCommand(self, action_idx):
        ''' Send the command to the game after a delay in a separate thread '''
        if action_idx not in [None, 3]:
            command_sender = CommandSenderGame(action_idx)
            self.threadpool.start(command_sender)


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
