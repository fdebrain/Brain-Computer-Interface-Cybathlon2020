from pathlib import Path

main_config = {
    'fs': 500,
    'n_channels': 63,
    'lsl_every_s': 0.1,
    'timeout': 0.5,
    'pred_decoding': {0: 'Rest', 1: 'Left', 2: 'Right', 3: 'Headlight'},
    'data_path': Path('../Datasets/Pilots'),
    'record_path': Path('./saved_recordings'),
    'models_path': Path('./saved_models'),
    'micro_path': Path('/dev/ttyACM*'),
}

train_config = {
    'n_crops': 10,
    'crop_len': 0.5,
    'f_order': 4,
    'f_low': 4,
    'f_high': 40,
    'n_jobs': 4         # -1 to use all available cpus, if out of RAM reduce the value
}

test_config = {
    'prefilt': False,
    'n_crops': 10,
    'crop_len': 0.5,
    'markers_decoding': {2: 'Rest', 4: 'Left', 6: 'Right', 8: 'Headlight'}
}

warmup_config = {
    'static_folder': Path('code/static'),
    'action2image': {'Left': 'arrow-left-solid.png',
                     'Right': 'arrow-right-solid.png',
                     'Rest': 'retweet-solid.png',
                     'Headlight': 'bolt-solid.png'},
    'record_name': 'warmup_recording.h5',
}

game_config = {
    'player_idx': 1,
    'game_path': Path('../game/brainDriver'),
    'game_logs_path': Path('../game/log'),
    'game_logs_pattern': 'raceLog*.txt',
    'fake_delay_min': 0,
    'fake_delay_max': 0,
    'record_name': 'game_recording.h5',
}

predictor_config = {
    # Corresponds to Fp1 and Fp2 (please double check)
    'ch_to_delete': [0, 30],
    'should_reref': True,
    'should_filter': False,
    'should_standardize': True,
    'select_last_s': 1,
    'n_crops': 10,
    'crop_len': 0.5,
    'predict_every_s': 1,
    'apply_notch': True,
    'apply_filt': False,
    'f_min': 2,
    'f_max': 40,
}
