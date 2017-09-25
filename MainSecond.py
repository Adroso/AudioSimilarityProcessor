"""
Program Created by Adrian Lapico ("Adroso") 25/9/17

Purpose: CP3000 Audio Processing Research Project - Second Attempt Starting From Scratch
Scope: Be able to classify samples of music into genre
Future Implementation: Be able to infer intended emotional response from music to make recommendations
"""

import matplotlib.style as ms
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np


def main():
    SONGS = ['Trainer_MusicType1/rock2.mp3', 'Trainer_MusicType2/tech1.mp3']
    SONGTYPES = ['rock', 'techno']
    raw_waveforms = audio_load(SONGS)
    plot_waves(SONGTYPES, raw_waveforms)


def audio_load(song_paths):
    """Loads via librosa all song within a passed in list of song paths"""
    loaded_songs = []
    for song in song_paths:
        try:
            y, sr = librosa.load(song)
            print("Audio {:10} Waveform Loaded".format(song))
            # loaded_songs.append([y, sr])
            loaded_songs.append(y)
        except:
            print("Failed To Load Audio File")
            print("Closing")

    return loaded_songs


def plot_waves(genres, raw_sounds):
    pass


main()
