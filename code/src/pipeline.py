import logging
import os
import json
import pickle

from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from .preprocessing import get_preprocessor
from .models import get_model


def get_pipeline(selected_preproc, preproc_config, model_config):
    preproc_steps, search_space_preproc = get_preprocessor(selected_preproc,
                                                           preproc_config)
    model, search_space_model = get_model(model_config)

    pipeline = Pipeline(steps=preproc_steps + [('model', model)])
    search_space = {**search_space_preproc,
                    **{'model__' + k: v
                       for k, v in search_space_model.items()}}
    return pipeline, search_space


def load_pipeline(model_path):
    """Load pre-trained model {FBCSP, Riemann, ShallowConvNet)

    Arguments:
        model_path {Path} -- Model path to load

    Raises:
        ValueError: Model suffix not recognised

    Returns:
        model {object} -- Model
    """
    logging.info(f'Loading model {model_path}')

    if model_path.suffix == '.pkl':
        unpickle = open(model_path, 'rb')
        model = pickle.load(unpickle)
    elif model_path.suffix == '.h5':
        model = load_model(str(model_path))
        logging.info(model.summary())
    else:
        raise ValueError('Model format not recognized !')

    logging.info(f'Successfully loaded model: {model}')
    return model


def save_pipeline(model, save_path, filename):
    """Save model as .pkl file.

    Arguments:
        model {object} -- Fitted model
        save_path {Path} -- Path of save directory
        filename {str} -- Name of file to save
    """
    if not os.path.isdir(save_path):
        logging.info(f'Creating directory {save_path}')
        os.mkdir(save_path)

    logging.info(f'Saving model {filename}.pkl')
    with open(save_path / f'{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info('Successfully saved model !')


def save_json(model_info, save_path, filename):
    with open(save_path / f'{filename}.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)
