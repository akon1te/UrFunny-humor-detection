import pickle
from pickle_loader import load_pickle

import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from moviepy.editor import *

from typing import Tuple, List, Dict

VIDEO_PATH = Path('../data') / 'urfunny2_video'
AUDIO_PATH = Path('../data') / 'urfunny2_audio'
DATA_PATH = Path('../data/')


class HumorDataset(Dataset):

    def __init__(self):

        self.origin_text = load_pickle(DATA_PATH / "language_sdk.pkl")
        self.preprocessed_text = []
        for idx in self.origin_text:
            language_feats = self.origin_text[idx]['context_sentences'] + \
                [self.origin_text[idx]['punchline_sentence']]
            language_feats = list(
                map(lambda i: ' ' + language_feats[i], range(0, len(language_feats))))
            language_feats = '.'.join(language_feats)
            self.preprocessed_text.append(language_feats[1::])

        target_dict = load_pickle(DATA_PATH / "humor_label_sdk.pkl")
        self.video = [VIDEO_PATH / f'{idx}.mp4' for idx in target_dict.keys()]
        self.audio = [AUDIO_PATH / f'{idx}.mp3' for idx in target_dict.keys()]
        self.files_idx = list(target_dict.keys())
        self.target = list(target_dict.values())

    def save(self):
        with open(DATA_PATH / 'text_sdk.pkl', 'wb') as t_file:
            pickle.dump(self.preprocessed_text, t_file)

        with open(DATA_PATH / 'video_sdk.pkl', 'wb') as v_file:
            pickle.dump(self.video, v_file)

        with open(DATA_PATH / 'audio_sdk.pkl', 'wb') as a_file:
            pickle.dump(self.audio, a_file)

    def __len__(self):
        return len(self.language_feats)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.files_idx[index], self.preprocessed_text[index], self.video[index], self.audio[index], self.target[index]
