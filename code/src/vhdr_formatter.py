from collections import Counter
import logging
import os

import numpy as np
import mne


def load_vhdr(vhdr_path, resample=False, preprocess=False, remove_ch=None):
    logging.info(f'Loading {vhdr_path}')

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True,
                                      verbose=False)

    if len(remove_ch) > 0:
        logging.info(f'Removing the following channels: {remove_ch}')
        raw.drop_channels(remove_ch)

    if resample:
        logging.info(f"Resampling from {raw.info['sfreq']} to 250 Hz")
        raw.resample(250)

    if preprocess:
        logging.info('Apply notch filter 50 Hz and '
                     'band-pass filtering [0.5, 100] Hz')
        raw = raw.notch_filter(freqs=[50])
        raw = raw.filter(l_freq=0.5, h_freq=100)
    return raw


def extract_events(raw, pre, post, marker_decodings, is_game=False):
    fs = raw.info['sfreq']
    to_delete = []

    # Extract events in format (ts, ?, marker_id)
    events, _ = mne.events_from_annotations(raw, verbose=False)

    # Game session - Keep markers corresponding to change in expected action + ignore model predictions
    if is_game:
        for idx in range(len(events) - 1):
            marker_id = str(events[idx][-1])

            # Ignore marker_id with less/more than 2 digits
            if len(marker_id) != 2:
                logging.info(f'Ignore event {events[idx][-1]}')
                to_delete.append(idx)
                continue

            # Remove event if no change in groundtruth
            next_marker_id = str(events[idx + 1][-1])
            if marker_id[0] == next_marker_id[0]:
                to_delete.append(idx + 1)

            # Replace marker_id by groundtruth (remove prediction)
            events[idx][-1] = int(marker_id[0])

        events = np.delete(events, to_delete, axis=0)
        to_delete = []

    # Getting rid of rest markers close to a consecutive one (less than 3.5s)
    for idx in range(len(events)-1):
        delay = (events[idx+1][0] - events[idx][0]) / fs
        if delay < 3.5 and events[idx][-1] == marker_decodings['Rest']:
            to_delete.append(idx)

    if len(to_delete) > 0:
        events = np.delete(events, to_delete, axis=0)
    logging.info(f'Removed {len(to_delete)} false events')
    logging.info([(e[0]/fs, e[-1]) for e in events])

    logging.info(f'Extracting window around markers: {-pre}s to {post}s')
    epochs = mne.Epochs(raw, events, marker_decodings, -pre, post,
                        baseline=None, preload=True, verbose=False)

    eeg = epochs.get_data()
    labels = epochs.events[:, -1]
    return eeg, labels


def format_session(list_paths, save_path, extraction_settings, preprocess_settings, marker_encodings, is_game):
    session_eeg, session_labels = [], []
    assert len(list_paths) > 0, 'No subsession to format'
    logging.info(f'Decoding: {extraction_settings["marker_decodings"]}')
    for subsession_path in list_paths:
        raw = load_vhdr(subsession_path, **preprocess_settings)
        eeg, labels = extract_events(raw, **extraction_settings,
                                     is_game=is_game)

        # Stack trials
        if len(session_eeg) == 0:
            session_eeg = eeg
            session_labels = labels
        else:
            session_eeg = np.vstack([session_eeg, eeg])
            session_labels = np.concatenate([session_labels, labels])

    # Remap labels
    session_labels = np.array([marker_encodings[l] for l in session_labels])
    logging.info(f'Encoding: {marker_encodings}')
    logging.info(f'Output labels: {Counter(session_labels)}')

    # Save as .npy file
    info = raw.info
    save_folder = f'formatted'
    save_folder += '_filt' if preprocess_settings['preprocess'] else '_raw'
    save_folder += f"_{int(info['sfreq'])}Hz"
    save_folder += '_game' if is_game else ''
    save_session(session_eeg, session_labels, info, save_path, save_folder)


def save_session(eeg, labels, info, save_path, save_folder):
    if save_folder not in os.listdir(save_path):
        logging.info(f'Creating folder {save_path}/{save_folder}')
        os.mkdir(f'{save_path}/{save_folder}')
        os.mkdir(f'{save_path}/{save_folder}/train')

    logging.info(f'Saving X: {eeg.shape} - y: {labels.shape}')
    filename = f'{save_path}/{save_folder}/train/train1.npz'
    np.savez_compressed(filename, X=eeg, y=labels,
                        fs=info['sfreq'], ch_names=info['ch_names'])
    logging.info('Successfully saved !')
