import pickle

import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from torch.utils.data import Dataset

from typing import Tuple, List, Dict

VIDEO_PATH = Path('../data') / 'urfunny2_video'
AUDIO_PATH = Path('../data') / 'urfunny2_audio'
DATA_PATH = Path('../data/')


def load_pickle(pickle_file) -> Dict:
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def metrics(y_pred, y_true):

    print(f"Accuracy: {accuracy_score(y_true, y_pred)}, \
        Precision: {precision_score(y_true, y_pred)}, \
            Recall: {recall_score(y_true, y_pred)}. \n\
            {confusion_matrix(y_true, y_pred)}")
