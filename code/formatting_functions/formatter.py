from preprocessing_functions.preproc_functions import filtering
import numpy as np
import glob
import os
import scipy.io
import scipy.signal
import re
import mne
from joblib import Parallel, delayed
import resampy
from copy import deepcopy
import h5py
import logging

import sys
sys.path.append('..')


class Formatter:
    '''
    Base class for dataset formatter.

    What this class does:
    - Import training/testing EEG data from npz/gdf/vhdr/mat format files.
    - Extract labels around each stimulus given parameters.
    - Save dataset in a compressed .npz format.

    Outputs:
    - X: EEG channels data | X.shape = (n_channels, n_samples)
    - y: Labels for each timestep | y.shape = (n_samples,)

    Parameters:
    - root: Folder containing the data
    - labels_idx: Stimulus encoding in the input data.
    - ch_list: List of channels names to pick (uses MNE -> only for gdf/vhdr).
    - remove_ch: Remove given channel names (uses MNE -> only for gdf/vhdr).
    - fs: Sampling frequency of the input data.
    - pre/post: Region of interest around each stimulus in seconds.
    - save: Save the returned dataset.
    - mode: Dataset folder to consider ('train'/'test').
    - save_path: Path where to save data.
    - save_folder: Name of save folder.
    - unknown_idx: Index corresponding to unknown event (used during labelling of test data)
    '''

    def __init__(self, root, save_path, labels_idx, ch_list, remove_ch, pre, post, mode, save, save_folder, resample=False, preprocess=False, unknown_idx=4, fs=250, save_as_trial=False):
        self.root = root
        self.labels_idx = labels_idx
        self.ch_list = ch_list
        self.remove_ch = remove_ch
        self.fs = fs
        self.pre = pre
        self.post = post
        self.save = save
        self.mode = mode
        self.save_path = save_path
        self.save_folder = save_folder
        self.unknown_idx = unknown_idx
        self.resample = resample
        self.preprocess = preprocess
        self.save_as_trial = save_as_trial

    def extracting(self, raw):
        # Resampling
        if self.resample:
            print('Resampling from {} Hz to 250 Hz'.format(raw.info['sfreq']))
            raw.resample(250, npad="auto")

        # Extract labels array
        events = mne.events_from_annotations(raw)[0]
        print('Detected labels: ', np.unique(events[:, -1]))

        # Extract sample frequency
        self.fs = raw.info['sfreq']
        print('Sampling frequency: ', self.fs)

        # Extract mapping
        mapping = mne.events_from_annotations(raw)[1]
        for key, val in mapping.items():
            print(key, val)

        # Extract channels
        if self.remove_ch:
            print('Removing the following channels: ', self.remove_ch)
            raw.drop_channels(self.remove_ch)
        if self.ch_list:
            print('Extracting the following channels: ', self.ch_list)
            raw.pick_channels(self.ch_list)

        # Preprocessing
        print('Numerical stability & filtering...')
        raw = mne_apply(lambda a: a * 1e6, raw)
        if self.preprocess:
            raw = mne_apply(lambda a: filtering(a, f_low=0.5,
                                                f_high=100,
                                                fs=raw.info["sfreq"],
                                                f_order=5), raw)

        # Convert to numpy array
        eeg = raw.get_data(verbose=False)
        logging.info('Signal shape: ', eeg.shape)

        # Reshape from (n_samples, n_channels) to (n_channels, n_samples)
        return eeg, events, mapping

    def resampling(self, raw, events, new_fs):
        """
        Resample continuous recording using `resampy`.
        Parameters
        """
        print('Resampling...')
        if new_fs == raw.info["sfreq"]:
            print("Just copying data, no resampling, since new sampling rate same.")
            return deepcopy(raw)
        print("Resampling from {:f} to {:f} Hz.".format(
            raw.info["sfreq"], new_fs))

        data = raw.get_data().T

        new_data = resampy.resample(data, raw.info["sfreq"],
                                    new_fs, axis=0, filter="kaiser_fast").T
        old_fs = raw.info["sfreq"]
        new_info = deepcopy(raw.info)
        new_info["sfreq"] = new_fs
        event_samples_old = events[:, 0]
        event_samples = event_samples_old * new_fs / float(old_fs)
        events[:, 0] = event_samples
        self.fs = new_fs
        return mne.io.RawArray(new_data, new_info), events

    def labelling(self, events, signal_len, file_idx):
        print("Counting labels before labelling")
        for label in np.unique(events[:, -1]):
            print("\t {}: {:.2f}".format(label, len(
                np.where(events[:, -1] == label)[0])))

        # Extract labels
        if self.mode == "train":
            labels = self.labelling_train(events, signal_len)

        elif self.mode == "test":
            data = scipy.io.loadmat(
                self.root + 'test/labels/{}.mat'.format(file_idx))
            true_labels = np.array(
                data['classlabel']) + min(self.labels_idx) - 1
            print('True labels set: ', np.unique(true_labels))
            labels = self.labelling_test(events, signal_len,
                                         true_labels, self.unknown_idx)

        # Sanity check
        print("Counting labels after labelling")
        for label in self.labels_idx:
            print("\t {}: {:.2f}".format(label, len(
                np.where(labels == label)[0]) / (self.fs*(self.pre + self.post))))

        return labels

    def labelling_train(self, events, signal_len):
        labels = np.zeros(signal_len, dtype=np.int16)

        # Convert window boundaries from s to nbr of time samples
        thresh_pre = int(self.pre*self.fs)
        thresh_post = int(self.post*self.fs)

        # Extract [pre,post] time window around each event
        for t, _, label in events:
            if label in self.labels_idx:
                labels[t-thresh_pre:t+thresh_post] = label

        print("Finished labelling...")
        return labels

    def labelling_test(self, events, signal_len, true_labels, event_id):
        labels = np.zeros(signal_len, dtype=np.int16)
        cnt = 0

        # Convert window boundaries from s to nbr of time samples
        thresh_pre = int(self.pre*self.fs)
        thresh_post = int(self.post*self.fs)

        # Extract [pre,post] time window around each event
        for t, _, label in events:
            if label == event_id:
                labels[t-thresh_pre:t+thresh_post] = true_labels[cnt]
                cnt += 1

        print("Finished labelling...")
        return labels

    def cutting(self, eeg, labels_in):
        n_channels, n_samples_in = eeg.shape
        trials = []
        labels = []
        trial = np.empty((n_channels, 0))

        for label in self.labels_idx:
            print("Processing trials of class {}...".format(label))
            idxs = np.where(labels_in == label)[0]
            trial = np.empty((n_channels, 0))

            # Go through each corresponding index, append eeg[idxs[i]] to current trial until idxs[i+1]-idx[i] > thresh
            for i in range(len(idxs)):
                if (idxs[i] - idxs[i-1]) > self.fs/20:  # Detect new trial
                    trials.append(trial)
                    labels.append(label)
                    trial = np.empty((n_channels, 0))

                sample = np.reshape(eeg[:, idxs[i]], (n_channels, 1))
                trial = np.append(trial, sample, axis=1)

                if i == len(idxs)-1:  # Detect end of last trial
                    trials.append(trial)
                    labels.append(label)

        trials = np.reshape(trials, (-1, n_channels, trial.shape[-1]))
        labels = np.reshape(labels, (-1,))

        print("Cutting successful ! Shape: {}".format(trials.shape))
        return trials, labels

    def saving(self, X, y=None, file_name='data1'):
        if self.save_folder not in os.listdir(self.save_path):
            os.mkdir(path=self.save_path + self.save_folder)
        if self.mode not in os.listdir(self.save_path + self.save_folder):
            os.mkdir(path=self.save_path + self.save_folder +
                     '/{}'.format(self.mode))

        # Remap labels
        y = y - np.min(y)

        # Reorder indexing
        idx = np.argsort(y)
        X, y = X[idx], y[idx]

        if self.save_as_trial:
            for trial_idx in range(X.shape[0]):
                np.savez_compressed(self.save_path + '{}/{}/{}'.format(
                    self.save_folder, self.mode, trial_idx), X=X[trial_idx], y=y[trial_idx])
        else:
            print("Saving data of shape: ", X.shape)
            np.savez_compressed(
                self.save_path + '{}/{}/{}'.format(self.save_folder, self.mode, file_name), X=X, y=y)
        print('Saved successfully ! \n')


class FormatterNPZ(Formatter):
    def __init__(self, root, save_path, labels_idx, ch_list, remove_ch, pre, post, fs, mode, save, save_folder):
        super().__init__(root, save_path, labels_idx, ch_list,
                         remove_ch, pre, post, mode, save, save_folder, fs=fs)
        self.unknown_idx = 783

    def run(self):
        # Get all filepaths for each pilot data
        if self.mode == "train":
            filepaths = glob.glob(self.root + "train/*T.npz")
        else:
            filepaths = glob.glob(self.root + "test/*E.npz")

        for pilot_idx, filepath in enumerate(filepaths):
            print("Start formatting " + filepath + "...")

            # Extract EEG signal
            raw = np.load(filepath)

            # Extract events & reshape EEG signal from (n_samples, n_channels) to (n_channels, n_samples)
            eeg = raw['s']
            events = np.stack(
                [raw['epos'].reshape(-1), raw['edur'].reshape(-1), raw['etyp'].reshape(-1)], axis=1)
            eeg = np.swapaxes(eeg, 0, 1)

            # Labelling
            to_find = r'A\d+T|B\d+T' if self.mode == 'train' else r'A\d+E|B\d+E'
            file_idx = re.findall(to_find, filepath)[0]
            print(file_idx)
            labels = self.labelling(events, eeg.shape[-1], file_idx)

            # Cutting
            eeg, labels = self.cutting(eeg, labels)

            # Save as .npy file
            if self.save:
                self.saving(eeg, labels, '{}{}'.format(self.mode, pilot_idx+1))
            print("")

        return print("Successfully converted all {} ".format(len(filepaths)) + self.mode + "ing data !")


class FormatterGDF(Formatter):
    def __init__(self, root, save_path, labels_idx, ch_list, remove_ch, pre, post, mode, save, save_folder, resample=False, preprocess=False, unknown_idx=4, fs=None, save_as_trial=False, multisession=False, multithread=False):
        super().__init__(root, save_path, labels_idx, ch_list, remove_ch, pre, post,
                         mode, save, save_folder, resample, preprocess, unknown_idx, fs, save_as_trial)
        self.labels_idx_saved = labels_idx
        self.multisession = multisession
        self.multithread = multithread

    def run(self):
        # Get all filepaths for each pilot data
        if self.mode == "train":
            filepaths_all = glob.glob(self.root + "train/*T.gdf")
        elif self.mode == "test":
            filepaths_all = glob.glob(self.root + "test/*E.gdf")
        else:
            print("Please choose among these two modes {train, test}.")
            return

        # Clustering common pilot data
        if self.multisession:
            filepaths_pilots = [[]]
            pilot_idx = 0
            to_find = r'B\d+T' if self.mode == 'train' else r'B\d+E'

            # In case of dataset IV 2b, there are multiple training sessions per subject
            for filepath in filepaths_all:
                if int(re.findall(to_find, filepath)[0][2]) == pilot_idx:
                    pass
                else:
                    pilot_idx += 1
                    filepaths_pilots.append([])
                filepaths_pilots[pilot_idx].append(filepath)
            filepaths_pilots.pop(0)  # Remove []
            print(filepaths_pilots)
        else:
            filepaths_pilots = [[path] for path in filepaths_all]

        # Multi-threading allows for faster processing (only after debugging complete)
        if self.multithread:
            num_cores = -1  # Use all processors but one
            results = Parallel(n_jobs=num_cores, verbose=100)(delayed(self.sub_run)(
                pilot, filepaths_pilot) for pilot, filepaths_pilot in enumerate(filepaths_pilots))
        else:
            for pilot, filepaths_pilot in enumerate(filepaths_pilots):
                print('Pilot ', pilot+1)
                self.sub_run(pilot, filepaths_pilot)

        return print("Successfully converted all {} ".format(len(filepaths_pilots)) + self.mode + "ing data !")

    def sub_run(self, pilot_idx, filepaths_pilot):
        pilot_eeg = np.array([])
        pilot_labels = np.array([])
        to_find = r'A\d+T|B\d+T' if self.mode == 'train' else r'A\d+E|B\d+E'

        for session_idx in range(len(filepaths_pilot)):
            print("Start formatting " + filepaths_pilot[session_idx] + "...")
            print('Session {}'.format(session_idx+1))

            # Quick fix for session 3 of dataset BCIC IV 2b
            if session_idx == 2:
                self.labels_idx_saved = self.labels_idx
                self.labels_idx = [4, 5]
            else:
                self.labels_idx = self.labels_idx_saved

            file_idx = re.findall(to_find, filepaths_pilot[session_idx])[0]

            # Import raw EEG signal (should be a .gdf file)
            raw = mne.io.read_raw_gdf(
                filepaths_pilot[session_idx], preload=True)

            # Extract eeg, events and map encodings
            eeg, events, mapping = self.extracting(raw)

            # Labelling
            labels = self.labelling(events, eeg.shape[-1], file_idx)

            # Cutting trials
            eeg, labels = self.cutting(eeg, labels)

            # Remap labels
            labels = labels - np.min(labels)
            print('Output labels', np.unique(labels))

            # Stack trials
            if len(pilot_eeg) == 0:
                pilot_eeg = eeg
                pilot_labels = labels
            else:
                pilot_eeg = np.vstack([pilot_eeg, eeg])
                pilot_labels = np.concatenate([pilot_labels, labels])

        # Save as .npy file
        if self.save:
            self.saving(pilot_eeg, pilot_labels,
                        '{}{}'.format(self.mode, pilot_idx+1))
        print("")


class FormatterVHDR(Formatter):
    def __init__(self, root, save_path, pilot_idx, session_idx, labels_idx, ch_list, remove_ch, pre, post, mode, save, save_folder, resample=False, preprocess=False, unknown_idx=4, control=False, save_as_trial=False):
        super().__init__(root, save_path, labels_idx, ch_list, remove_ch, pre, post, mode, save,
                         save_folder, resample, preprocess, unknown_idx, fs=None, save_as_trial=save_as_trial)

        self.pilot_idx = pilot_idx
        self.session_idx = session_idx
        self.control = control

    def run(self):
        pilot_eeg = np.array([])
        pilot_labels = np.array([])

        if self.control:
            filepaths_pilot = glob.glob(
                f'{self.root}Control_{self.pilot_idx}/Session_{self.session_idx}/vhdr/*.vhdr')
            self.save_path = f'{self.root}Control_{self.pilot_idx}/Session_{self.session_idx}/'
        else:
            filepaths_pilot = glob.glob(self.root + 'Pilot_{}/Session_{}/vhdr/{}*.vhdr'.format(
                self.pilot_idx, self.session_idx, 'test/' if self.mode == 'test/' else ''))
            self.save_path = f'{self.root}Pilot_{self.pilot_idx}/Session_{self.session_idx}/'

        logging.info(f'List of EEG sessions: {filepaths_pilot}')
        assert len(filepaths_pilot), 'VHDR file not found !'

        # Gather all subsessions into one
        for sub_session_idx in range(len(filepaths_pilot)):
            logging.info(
                f'Start formatting {filepaths_pilot[sub_session_idx]} ...')
            logging.info(f'Sub-session {sub_session_idx + 1}')

            # Import raw EEG file
            raw = mne.io.read_raw_brainvision(filepaths_pilot[sub_session_idx],
                                              preload=True, verbose=False)
            # Resample
            if self.resample:
                logging.info(f"Resampling from {raw.info['sfreq']} to 250 Hz")
                raw.resample(250)

            # Extract events
            events = mne.events_from_annotations(raw, verbose=False)[0]

            # Extract channels
            if self.remove_ch:
                logging.info(
                    f'Removing the following channels: {self.remove_ch}')
                raw.drop_channels(self.remove_ch)
            if self.ch_list:
                logging.info(
                    f'Extracting the following channels: {self.ch_list}')
                raw.pick_channels(self.ch_list)

            logging.info('Fixing numerical unstability')
            raw = mne_apply(lambda a: a * 1e6, raw)

            # Filtering
            if self.preprocess:
                logging.info('Filtering')
                raw = mne_apply(lambda a: filtering(a, f_low=0.5,
                                                    f_high=100,
                                                    fs=raw.info['sfreq'],
                                                    f_order=5), raw)

            event_ids = dict(zip(['left', 'right', 'both', 'rest'],
                                 self.labels_idx))

            logging.info(
                f'Extracting epochs in event frame (=0s) {-self.pre}s to {self.post}s')
            epochs = mne.Epochs(raw, events, event_ids, -self.pre,
                                self.post, baseline=None, preload=True,
                                verbose=False)

            eeg = epochs.get_data()
            labels = epochs.events[:, -1]

            # Stack trials
            if len(pilot_eeg) == 0:
                pilot_eeg = eeg
                pilot_labels = labels
            else:
                pilot_eeg = np.vstack([pilot_eeg, eeg])
                pilot_labels = np.concatenate([pilot_labels, labels])

        # Remap labels
        pilot_labels = pilot_labels - np.min(pilot_labels)
        logging.info(f'Output labels: {np.unique(pilot_labels)}')

        # Save as .npy file
        if self.save:
            self.saving(pilot_eeg, pilot_labels, '{}1'.format(self.mode))


class FormatterMAT(Formatter):
    def __init__(self, root, save_path, labels_idx, ch_list, remove_ch, pre, post, mode, save, save_folder, resample=False, preprocess=False, unknown_idx=4, fs=None, save_as_trial=False, multisession=False, multithread=False):
        super().__init__(root, save_path, labels_idx, ch_list, remove_ch, pre, post,
                         mode, save, save_folder, resample, preprocess, unknown_idx, fs, save_as_trial)

    def run(self):
        # Get all filepaths for each pilot data
        if self.mode == "train":
            filepaths_all = glob.glob(self.root + "train/*.mat")
        elif self.mode == "test":
            filepaths_all = glob.glob(self.root + "test/*.mat")
        else:
            print("Please choose among these two modes {train, test}.")
            return

        for pilot_idx in range(len(filepaths_all)):
            # Load .mat file
            data = h5py.File(filepaths_all[pilot_idx], "r")

            # Extract some info
            self.fs = data['nfo']['fs'][0, 0]
            n_samples = int(data['nfo']['T'][0, 0])
            ch_pos = data['mnt']['pos_3d'][:]
            print('fs: {} - n_samples: {}'.format(self.fs, n_samples))

            # Get class names
            class_name_set = data["mrk"]["className"][:].squeeze()
            all_class_names = ["".join(chr(c) for c in data[obj_ref])
                               for obj_ref in class_name_set]
            print('Classes: ', all_class_names)

            # Extract eeg
            ch_names = ['ch{}'.format(c) for c in range(1, 134)]
            eeg = []
            eeg = np.array([data[ch_name][:].squeeze()
                            for ch_name in ch_names])
            print(eeg.shape)

            # Extract events
            events_c = data['mrk']['event']['desc'][:
                                                    ].reshape(-1, 1).astype(int)
            events_t = data['mrk']['time'][:].reshape(-1, 1).astype(int)
            events = np.hstack([events_t, events_c, events_c])

            # Labelling
            assert np.allclose(np.unique(events_c), np.unique(self.labels_idx))
            labels = self.labelling_train(events, n_samples)
            print(eeg.shape, labels.shape)

            # Cutting
            eeg, labels = self.cutting(eeg, labels)
            print(eeg.shape, labels.shape)

            # Save as .npy file
            if self.save:
                self.saving(eeg, labels, '{}{}'.format(self.mode, pilot_idx+1))
                print("")


def mne_apply(func, raw):
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=False)
