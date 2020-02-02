import logging
import serial
import socket

from pyqtgraph.Qt import QtCore


class GamePlayer:
    def __init__(self, player_idx):
        # Send commands using a separate thread
        self.thread_game = QtCore.QThreadPool()

    def sendCommand(self, action_idx):
        ''' Send the command to the game after a delay in a separate thread '''
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
        self.commands = {0: "\x0B", 1: "\x0D", 2: "\x0C"}

    @QtCore.pyqtSlot()
    def run(self):
        assert self.action_idx in self.commands.keys(), 'Action unknown'
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
