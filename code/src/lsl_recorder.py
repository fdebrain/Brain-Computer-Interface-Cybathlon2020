import os
import logging
import h5py
import numpy as np
# from pyqtgraph.Qt import QtCore

from config import main_config


class LSLRecorder:
    def __init__(self, recording_folder, filename, debug=False):
        self.fs = main_config['fs']
        self.n_channels = main_config['n_channels']

        self.h5_path = recording_folder / filename
        if os.path.isfile(self.h5_path):
            self.h5_path = recording_folder / \
                f'{filename[:-3]}_{np.random.randint(0, 9999)}.h5'
            logging.info(self.h5_path)
        self.debug = debug

    def open_h5(self):
        logging.info(f'Start recording in {self.h5_path}')
        self.h5 = h5py.File(self.h5_path, 'a')
        self.ts_set = self.h5.create_dataset(name='ts',
                                             shape=(0,),
                                             maxshape=(None,),
                                             dtype=np.float32,
                                             chunks=(self.fs,))
        self.eeg_set = self.h5.create_dataset(name='eeg',
                                              shape=(self.n_channels, 0,),
                                              maxshape=(self.n_channels,
                                                        None,),
                                              dtype=np.float32,
                                              chunks=(self.n_channels,
                                                      self.fs,))
        self.event_set = self.h5.create_dataset(name='event',
                                                shape=(0, 2),
                                                maxshape=(None, 2),
                                                dtype=np.float32)

    def close_h5(self):
        logging.info('Stop recording')
        self.h5.close()

    def save_event(self, ts, event):
        self.event_set.resize(self.event_set.shape[0] + 1, axis=0)
        self.event_set[-1:, :] = [ts, event]

        if self.debug:
            logging.info(f'ts:{ts} - action: {event}')

    def save_data(self, ts, eeg):
        if ts is not None or eeg is not None:
            n_frames = len(ts)
            self.eeg_set.resize(self.eeg_set.shape[1] + n_frames, axis=1)
            self.eeg_set[:, -n_frames:] = eeg

            self.ts_set.resize(self.ts_set.shape[0] + n_frames, axis=0)
            self.ts_set[-n_frames:] = ts

            if self.debug:
                logging.info(
                    f'ts: {self.ts_set.shape} - eeg: {self.eeg_set.shape} - n_frames: {n_frames}')
