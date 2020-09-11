import numpy as np
import time
from src.lsl_client import LSLClient
from src.lsl_recorder import LSLRecorder

if __name__ == '__main__':
    lsl_recorder = LSLRecorder(h5_name='data.h5')
    lsl_reader = LSLClient()

    countdown = time.time()
    while time.time() - countdown < 3:
        if time.time() - countdown > 1:
            eeg, ts = None, None

            try:
                eeg, ts = lsl_reader.get_data()
            except Exception as e:
                print(f'No more data - {e}')
                continue

            if len(eeg) > 0:
                eeg = np.swapaxes(eeg, 1, 0)
                lsl_recorder.save_data(eeg, ts)

            countdown = time.time()
