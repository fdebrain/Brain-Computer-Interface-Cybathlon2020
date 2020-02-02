import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams


class LSLClient:
    def __init__(self):
        logging.info('Looking for LSL stream...')
        available_streams = resolve_streams(5)

        if len(available_streams) > 0:
            self.stream_reader = StreamInlet(available_streams[0],
                                             max_chunklen=1,
                                             recover=False)
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
            # Data is of shape (n_timestamps, n_channels)
            data, ts = self.stream_reader.pull_chunk()
        except Exception as e:
            logging.info(f'{e} - No more data')
        return np.array(data), np.array(ts)
