import mne
from mne_realtime import MockLSLStream
import time
import logging

stream_file = '../Datasets/Pilots/Pilot_2/Session_18/game/EEG64_CY_pilot_07032002_18_12_game1.vhdr'


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

    with MockLSLStream(host, raw, 'eeg', time_dilation=1) as streamer:
        counter = 0
        countdown = time.time()
        logging.info(
            f'Real-time streaming from EEG recording at {streamer._sfreq} Hz')

        while counter < raw.n_times / raw.info['sfreq']:
            elapsed_time = time.time() - countdown
            if elapsed_time > 1.:
                print(
                    f'Current timestamp: {streamer._sfreq*counter:.2f} - Elapsed: {elapsed_time:.2f}')
                counter += elapsed_time
                countdown = time.time()
