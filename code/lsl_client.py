from pylsl import StreamInlet, resolve_streams
import sys
import logging
import numpy as np

# Setup logging
root_logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

fs = 500
dt = []

if __name__ == '__main__':
    logging.info('Looking for LSL stream...')
    available_streams = resolve_streams(5)

    if len(available_streams) > 0:
        stream_reader = StreamInlet(available_streams[0], max_chunklen=1)
        logging.info(f'Stream {stream_reader.info().source_id()} found !')
    else:
        logging.error('No stream found !')
        sys.exit(0)

    data, t0 = stream_reader.pull_sample()
    logging.info(f'Timestamp: {t0} - Data: {data[0]}')
    tk = t0
    for _ in range(fs):
        data, ts = stream_reader.pull_sample()
        dt.append(ts - tk)
        tk = ts
    logging.info(f"Fetching 1s worth of data in: {(tk - t0):.2f}s")
    logging.info(
        f"Between consecutive samples: {np.mean(dt):.4f} + - {np.std(dt):.4f}")
    logging.info(f'Time dilation of {1./(tk-t0):.3f}')
    stream_reader.close_stream()
