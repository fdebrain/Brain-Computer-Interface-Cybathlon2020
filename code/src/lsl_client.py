import time
import copy
import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams
from pyqtgraph.Qt import QtCore

from config import main_config


class LSLClient(QtCore.QRunnable):
    def __init__(self, parent, debug=False):
        super().__init__()
        self.parent = parent
        self.ts, self.eeg = [], []
        self.fetch_every_s = main_config['lsl_every_s']
        self.chunk_len = int(main_config['fs'] * self.fetch_every_s)
        self.create_stream()
        self.should_stream = True
        self.debug = debug

    def create_stream(self):
        logging.info('Looking for LSL stream...')
        available_streams = resolve_streams(5)

        if len(available_streams) > 0:
            self.stream_reader = StreamInlet(available_streams[0],
                                             max_chunklen=self.chunk_len,
                                             recover=False)

            # Extract stream info
            id = self.stream_reader.info().session_id()
            self.fs = int(self.stream_reader.info().nominal_srate())
            self.n_channels = int(self.stream_reader.info().channel_count())
            logging.info(f'Stream {id} found at {self.fs} Hz '
                         f'with {self.n_channels} channels')

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
            # Fetch available data from lsl stream and convert to numpy array
            eeg, ts = self.stream_reader.pull_chunk(timeout=main_config['timeout'],
                                                    max_samples=self.chunk_len)
            self.eeg = np.array(eeg, dtype=np.float64)
            self.ts = np.array(ts, dtype=np.float64)

            if len(self.ts) < self.chunk_len:
                logging.info(f'Receiving LSL: {len(self.ts)} '
                             f'instead of {self.chunk_len} !')

        except Exception as e:
            logging.info(f'{e} - No more data')
            self.should_stream = False

    def notify(self):
        if len(self.eeg) > 0:
            # Manipulate eeg data to be of shape (n_channels, n_timestamps)
            self.eeg = np.swapaxes(self.eeg, 1, 0)

            # Convert timestamps from seconds to framesize
            self.ts = np.array(self.ts) * self.fs
            self.ts = self.ts.astype(np.int64)

            self.parent.lsl_data = (copy.deepcopy(self.ts),
                                    copy.deepcopy(self.eeg))

            if self.debug:
                logging.info(f'Receiving LSL - '
                             f'Last ts: {self.ts[-1]} - '
                             f'n_frames: {len(self.ts)}')

    @QtCore.pyqtSlot()
    def run(self):
        logging.info('Start LSL stream')
        while self.should_stream is True:
            countdown = time.time()
            self.get_data()
            self.notify()
            delay = time.time() - countdown

            if self.fetch_every_s - delay > 0:
                time.sleep(self.fetch_every_s - delay)
        logging.info('Stop LSL stream')
