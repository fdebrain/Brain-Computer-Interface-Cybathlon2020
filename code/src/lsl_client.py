import time
import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams
from pyqtgraph.Qt import QtCore


class LSLClient(QtCore.QRunnable):
    def __init__(self, parent, fetch_every_s=0.1):
        super().__init__()
        self.parent = parent
        self.fetch_every_s = fetch_every_s
        self.create_stream()
        self.should_stream = True

    def create_stream(self):
        logging.info('Looking for LSL stream...')
        available_streams = resolve_streams(5)

        if len(available_streams) > 0:
            self.stream_reader = StreamInlet(available_streams[0],
                                             max_chunklen=1,
                                             recover=False)

            # Extract stream info
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
            # Fetch available data from lsl stream and convert to numpy array
            data, ts = self.stream_reader.pull_chunk()
            self.data = np.array(data, dtype=np.float32)
            self.ts = np.array(ts)
        except Exception as e:
            logging.info(f'{e} - No more data')

    def notify(self):
        if len(self.data) > 0:
            # Manipulate data to be of shape (n_channels, n_timestamps)
            self.data = np.swapaxes(self.data, 1, 0)
            self.parent.lsl_data = self.data
            self.parent.lsl_ts = self.ts

    @QtCore.pyqtSlot()
    def run(self):
        while self.should_stream is True:
            self.get_data()
            self.notify()
            time.sleep(self.fetch_every_s)
