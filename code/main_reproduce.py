from offline_pipeline.pipeline_ml import MLExperiment


# Fixed params
folder_type = 'Pilot'
folder_format = 'formatted_filt_250Hz'
pilot_idx = 2
params = \
    {
        'path': '',
        'session_idx': -1,
        'model_name': 'FBCSP',
        'MI_labels': ['Rest', 'Left', 'Right', 'Both'],
        'fs': 250,
        'start': 2.5,
        'end': 6.,
        'filt': False,
        'rereferencing': False,
        'standardization': False,
        'n_crops': 1,
        'crop_len': 3.5,
        'average_pred': False,
        'get_online_metrics': False,
        'm': 2,
        'C': 10
    }

if __name__ == '__main__':
    for session_idx in [3, 4, 5, 6, 8]:
        params['session_idx'] = session_idx
        params['path'] = f'../Datasets/{folder_type}s/{folder_type}_{pilot_idx}/Session_{session_idx}/{folder_format}/'

        print(f'\n\nRUNNING EXP ON session {session_idx}')
        exp = MLExperiment(params)
        exp.run()
