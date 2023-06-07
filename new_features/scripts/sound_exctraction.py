from moviepy.editor import *
from pathlib import Path
from tools import load_pickle
from tqdm import tqdm


def exctraction():
    target_dict = load_pickle("../../data/humor_label_sdk.pkl")
    for idx in tqdm(target_dict.keys()):
        video = VideoFileClip(f'../../data/urfunny2_video/{idx}.mp4')
        video.audio.write_audiofile(f'../../data/urfunny2_audio/{idx}.mp3')


if __name__ == "__main__":
    exctraction()
