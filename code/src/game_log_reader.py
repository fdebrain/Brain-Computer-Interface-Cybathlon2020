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
            if ("start race" in line):
                self.notify((0, 'Game start'))

            # Exctract expected actions (groundtruth) & notify player widget
            elif ("p{}_expectedInput".format(self.player_idx) in line):
                if "none" in line:
                    expected_action = self.log2actions['none']
                else:
                    action = line.split(" ")[-1].strip()
                    expected_action = self.log2actions[action]

                logging.info(f"Groundtruth: {expected_action[1]}")
                self.notify(expected_action)
