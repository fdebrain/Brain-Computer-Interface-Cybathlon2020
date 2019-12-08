import mne
from mne_realtime import MockLSLStream
from online_pipeline.config import stream_file
import time
import logging

# Setup logging
root_logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)


if __name__ == '__main__':
    # Load raw data to stream
    raw = mne.io.read_raw_brainvision(stream_file, preload=True)

    # Stream data in real-time
    host = 'pilot_stream'
    streamer = MockLSLStream(host, raw, 'eeg', time_dilation=1)
    streamer.start()
    logging.info(
        f'Real-time streaming from EEG recording at {streamer._sfreq} Hz')

    time.sleep(50)

    streamer.stop()
