import logging
import copy
import time
import numpy as np
from pyqtgraph.Qt import QtCore

from src.pipeline import load_pipeline
from src.models import predict


class ActionPredictor(QtCore.QRunnable):
    def __init__(self, parent, modelfile, is_convnet, fs=500, predict_every_s=1):
        super(ActionPredictor, self).__init__()
        self.parent = parent
        self.fs = fs
        self.ch_to_delete = [0, 30]
        self.should_reref = True
        self.should_filter = False
        self.should_standardize = True
        self.model = load_pipeline(modelfile)
        self.is_convnet = is_convnet
        self.n_crops = 10
        self.crop_len = 0.5
        self.predict_every_s = predict_every_s
        self.action_idx = 0
        self.should_predict = True

    def predict(self, X):
        # Removing FP1 & FP2
        X = np.delete(X, self.ch_to_delete, axis=0)

        # Selecting last 1s of signal
        X = X[np.newaxis, :, -self.fs:]

        # If no model
        if self.model is None:
            self.action_idx = 0
            logging.warning('Rest action sent by default! '
                            'Please select a model.')
        else:
            self.action_idx = predict(X, self.model, self.is_convnet,
                                      self.n_crops, self.crop_len, self.fs,
                                      self.should_reref, self.should_filter,
                                      self.should_standardize)
            logging.info(f'Action idx: {self.action_idx}')

    def notify(self):
        self.parent.current_pred = self.action_idx

    @QtCore.pyqtSlot()
    def run(self):
        while self.should_predict is True:
            countdown = time.time()
            X = copy.deepcopy(self.parent.input_signal)
            self.predict(X)
            self.notify()
            delay = time.time() - countdown
            time.sleep(self.predict_every_s - delay)
