import logging
import copy
import time

import numpy as np
from pyqtgraph.Qt import QtCore
import mne

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
        if is_convnet:
            self.should_reref = predictor_config['should_reref']
            self.should_filter = predictor_config['should_filter']
            self.should_standardize = predictor_config['should_standardize']
        else:
            self.should_reref = False
            self.should_filter = False
            self.should_standardize = False
        self.n_crops = predictor_config['n_crops']
        self.crop_len = predictor_config['crop_len']
        self.apply_notch = predictor_config['apply_notch']
        self.apply_filt = predictor_config['apply_filt']
        self.f_min = predictor_config['f_min']
        self.f_max = predictor_config['f_max']

        # Prediction
        self.predict_every_s = predictor_config['predict_every_s']
        self.fake_delay_min = game_config['fake_delay_min']
        self.fake_delay_max = game_config['fake_delay_max']
        self.should_predict = True
        self.action_idx = 0
        self.select_last_s = predictor_config['select_last_s']

        # Model
        self.is_convnet = is_convnet
        if modelfile == 'AUTOPLAY':
            self.model = 'AUTOPLAY'
        else:
            self.model = load_pipeline(modelfile)

        # Game player
        self.player_idx = game_config['player_idx']
        self.game_player = GamePlayer(self.player_idx)
        self.game_logs_path = game_config['game_logs_path']

    @property
    def available_logs(self):
        logs = list(self.game_logs_path.glob(game_config['game_logs_pattern']))
        return sorted(logs)

    def preproc_signal(self, eeg):
        ch_names = np.delete(self.parent.lsl_reader.ch_names,
                             self.ch_to_delete, axis=0)

        info = mne.create_info(list(ch_names),
                               self.fs,
                               ch_types='eeg')

        logging.disable(logging.INFO)
        data = mne.io.RawArray(eeg, info, verbose=0)

        if self.apply_notch:
            data.notch_filter(freqs=[50])
        if self.apply_filt:
            data.filter(l_freq=self.f_min, h_freq=self.f_max)
        logging.disable(logging.NOTSET)

        return data.get_data()

    def predict(self, X):
        if self.model is None:
            self.action_idx = 0
            logging.warning('Rest action sent by default! '
                            'Please select a model.')
        elif self.model == 'AUTOPLAY':
            random_delay = (self.fake_delay_max - self.fake_delay_min) * \
                np.random.random_sample() + self.fake_delay_min
            if random_delay > 0:
                logging.info(f'Sleep for {random_delay}s')
                time.sleep(random_delay)
            self.action_idx = self.parent.expected_action[0]
        else:
            # Removing FP1 & FP2
            X = np.delete(X, self.ch_to_delete, axis=0)

            # Preprocess signal
            X = self.preproc_signal(X)

            # Selecting last 1s of signal
            X = X[np.newaxis, :, -int(self.fs*self.select_last_s):]

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
        self.parent.pred_action = (copy.deepcopy(self.action_idx),
                                   copy.deepcopy(self.pred_decoding[self.action_idx]))

    @QtCore.pyqtSlot()
    def run(self):
        logging.info('Start action predictor')
        while self.should_predict is True:
            countdown = time.time()
            X = copy.deepcopy(self.parent.input_signal)
            self.predict(X)
            self.notify()
            delay_correction = time.time() - countdown
            if delay_correction > 0:
                delay_correction = min(delay_correction, self.predict_every_s)
                time.sleep(self.predict_every_s - delay_correction)

        logging.info('Stop action predictor')
