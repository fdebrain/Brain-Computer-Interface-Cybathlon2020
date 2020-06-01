import logging
import time
from pyqtgraph.Qt import QtCore


class GameLogReader(QtCore.QRunnable):
    def __init__(self, parent, logfilename, player_idx):
        super(GameLogReader, self).__init__()
        self.parent = parent
        self.logfilename = logfilename
        self.player_idx = player_idx
        self.log2actions = {"none": (0, "Rest"),
                            "leftWinker": (1, "Left"),
                            "rightWinker": (2, "Right"),
                            "headlight": (3, "Light")}

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
            if "start race" in line:
                self.notify((0, 'Game start'))
            elif f"p{self.player_idx}_expectedInput" in line:
                # Exctract expected actions (groundtruth) & notify player widget
                if "none" in line:
                    expected_action = self.log2actions['none']
                else:
                    action = line.split(" ")[-1].strip()
                    expected_action = self.log2actions[action]
                logging.info(f"Groundtruth: {expected_action[1]}")
                self.notify(expected_action)
            elif f"p{self.player_idx}_finish" in line:
                # Detect if player finishes game
                self.notify((0, 'Game end'))
            elif "puase race" in line or "puase race" in line:
                # Detect pause (there is a typo in the game automatic logger)
                self.notify((0, 'Pause'))
            elif "resume paused race" in line:
                # Detect resume from pause
                self.notify((0, 'Resume'))
