from pathlib import Path

main_config = {
    'fs': 500,
    'n_channels': 63,
    'lsl_every_s': 0.1,
    'pred_decoding': {0: 'Rest', 1: 'Left', 2: 'Right', 3: 'Headlight'},
    'data_path': Path('../Datasets/Pilots'),
    'record_path': Path('./saved_recordings'),
    'models_path': Path('./saved_models'),
    'micro_path': Path('/dev/ttyACM*'),
}

test_config = {
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
    'fake_delay_min': 0.5,
    'fake_delay_max': 1,
    'record_name': 'game_recording.h5',
}

predictor_config = {
    'ch_to_delete': [0, 30],
    'should_reref': True,
    'should_filter': False,
    'should_standardize': True,
    'n_crops': 10,
    'crop_len': 0.5,
    'predict_every_s': 1,
}
