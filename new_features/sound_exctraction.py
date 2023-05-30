from moviepy.editor import *
from pathlib import Path
from pickle_loader import load_pickle

DATA_PATH = Path('../sdk_features/')


def exctraction():
    target_dict = load_pickle(DATA_PATH / "humor_label_sdk.pkl")
    for idx in target_dict.keys():
        video = VideoFileClip(f'../data/urfunny2_video/{idx}.mp4')
        video.audio.write_audiofile(f'../data/urfunny2_audio/{idx}.mp3')


if __name__ == "__main__":
    exctraction()
