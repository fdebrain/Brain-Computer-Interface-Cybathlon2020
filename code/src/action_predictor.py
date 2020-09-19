import logging
import copy
import time

import numpy as np
from pyqtgraph.Qt import QtCore

from config import main_config, predictor_config, game_config
from src.pipeline import load_pipeline
from src.models import predict
from src.game_player import GamePlayer


class ActionPredictor(QtCore.QRunnable):
    def __init__(self, parent, modelfile, is_convnet):
        super(ActionPredictor, self).__init__()
        self.parent = parent
        self.fs = main_config['fs']
        self.pred_decoding = main_config['pred_decoding']

        # Preprocessing
        self.ch_to_delete = predictor_config['ch_to_delete']
        self.should_reref = predictor_config['should_reref']
        self.should_filter = predictor_config['should_filter']
        self.should_standardize = predictor_config['should_standardize']
        self.n_crops = predictor_config['n_crops']
        self.crop_len = predictor_config['crop_len']

        # Prediction
        self.predict_every_s = predictor_config['predict_every_s']
        self.fake_delay_min = game_config['fake_delay_min']
        self.fake_delay_max = game_config['fake_delay_max']
        self.should_predict = True
        self.action_idx = 0

        # Model
        self.is_convnet = is_convnet
        if modelfile == 'AUTOPLAY':
            self.model = 'AUTOPLAY'
        else:
            self.model = load_pipeline(modelfile)

        # Game logs (separate thread - if autoplay)
        self.game_logs_path = game_config['game_logs_path']
        self.game_log_reader = None
        # self.thread_log = QtCore.QThreadPool()

        # Game player
        self.player_idx = game_config['player_idx']
        self.game_player = GamePlayer(self.player_idx)

    @property
    def available_logs(self):
        logs = list(self.game_logs_path.glob(game_config['game_logs_pattern']))
        return sorted(logs)

    def predict(self, X):
        # Removing FP1 & FP2
        X = np.delete(X, self.ch_to_delete, axis=0)

        # Selecting last 1s of signal
        X = X[np.newaxis, :, -self.fs:]

        if self.model is None:
            self.action_idx = 0
            logging.warning('Rest action sent by default! '
                            'Please select a model.')
        elif self.model == 'AUTOPLAY':
            self.action_idx = self.parent.expected_action[0]
        else:
            self.action_idx = predict(X, self.model, self.is_convnet,
                                      self.n_crops, self.crop_len, self.fs,
                                      self.should_reref, self.should_filter,
                                      self.should_standardize)
        logging.info(f'Action idx: {self.action_idx}')

        # Send action to avatar (if game is on + not rest command)
        possible_actions = [k for k in self.pred_decoding.keys()]
        is_action_command = self.action_idx in possible_actions \
            and self.pred_decoding[self.action_idx] != 'Rest'
        if self.parent.game_is_on and is_action_command:
            logging.info(f'Sending: {self.action_idx}')
            self.game_player.sendCommand(self.action_idx)

    def notify(self):
        # AUTOPLAY - Fake delay for more realistic control feel
        if self.model == 'AUTOPLAY' and self.action_idx in self.pred_decoding.keys():
            random_delay = (self.fake_delay_max - self.fake_delay_min) * \
                np.random.random_sample() + self.fake_delay_min
            time.sleep(random_delay)

        self.parent.pred_action = (copy.deepcopy(self.action_idx),
                                   copy.deepcopy(self.pred_decoding[self.action_idx]))

    @QtCore.pyqtSlot()
    def run(self):
        logging.info('Start action predictor')
        while self.should_predict is True:
            countdown = time.time()
            X = copy.deepcopy(self.parent.input_signal)
            self.predict(X)
            delay_correction = time.time() - countdown
            time.sleep(self.predict_every_s - delay_correction)
            self.notify()

        logging.info('Stop action predictor')
