import os
import h5py
import numpy as np
class LSLRecorder:
    def __init__(self, h5_name='test.h5', n_channels=63, fs=500, debug=False):
        if os.path.isfile(h5_name):
            h5_name += f'_{np.random.randint(0,9999)}'
        print(f'Start recording LSL stream in {h5_name}')
        self.h5 = h5py.File(h5_name, 'a')
        self.eeg_set = self.h5.create_dataset(name='eeg',
                                              shape=(n_channels, 2*fs,),
                                              maxshape=(n_channels, None,),
                                              dtype=np.float32,
                                              chunks=(n_channels, 2*fs,))
        self.ts_set = self.h5.create_dataset(name='ts',
                                             shape=(fs,),
                                             maxshape=(None,),
                                             dtype=np.float32,
                                             chunks=(fs,))
        self.debug = debug

    def __del__(self):
        self.h5.close()

    def save_data(self, eeg, ts):
        n_frames = eeg.shape[-1]
        self.eeg_set.resize(self.eeg_set.shape[1] + n_frames, axis=1)
        self.eeg_set[:, -n_frames:] = eeg

        self.ts_set.resize(self.ts_set.shape[0] + n_frames, axis=0)
        self.ts_set[-n_frames:] = ts

        if self.debug:
            print(
                f'ts: {self.ts_set.shape} - eeg: {self.eeg_set.shape} - Nb new samples: {n_frames}')
