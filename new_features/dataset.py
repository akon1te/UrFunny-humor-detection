import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader

from moviepy.editor import *

from pickle_loader import load_pickle

from typing import Tuple, List, Dict

VIDEO_PATH = Path('../data') / 'urfunny2_video'
AUDIO_PATH = Path('../data') / 'urfunny2_audio'
DATA_PATH = Path('../sdk_features/')

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
            self.preprocessed_text.append(language_feats)

        target_dict = load_pickle(DATA_PATH / "humor_label_sdk.pkl")
        self.videos = [VIDEO_PATH / f'{idx}.mp4' for idx in target_dict.keys()]
        self.audio = [AUDIO_PATH / f'{idx}.mp3' for idx in target_dict.keys()]
        self.target = list(target_dict.values())

    def __len__(self):
        return len(self.language_feats)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.preprocessed_text[index], self.videos[index], self.audio[index], self.target[index]