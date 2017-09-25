"""
Program Created by Adrian Lapico ("Adroso") 2/9/17

Purpose: CP3000 Audio Processing Research Project
Scope: Be able to classify samples of music into genre
Future Implementation: Be able to infer intended emotional response from music to make recommendations
"""

# Imports
import matplotlib.style as ms

ms.use('seaborn-muted')
import librosa
import librosa.display
import os
import numpy as np
from  sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=np.nan)
import tensorflow as tf

# filefinding
# from tkinter.filedialog import askopenfilename


TRAINING_1 = 'Trainer_MusicType1/'
TRAINING_2 = 'Trainer_MusicType2/'
TEST = "TestData/"


def main():
    print("Starting Program..")

    training_files_1 = directory_loader(TRAINING_1)
    # training_files_2 = directory_loader(TRAINING_2)
    # test_files = directory_loader(TEST)


    # Loading Each Directory of songs
    loaded_songs_1 = audio_load(training_files_1, 1)
    # loaded_songs_2 = audio_load(training_files_2, 2)
    # loaded_songs_3 = audio_load(test_files, 3)

    # Calculating each directories songs BPM
    # song_bpms_1 = find_bpm(loaded_songs_1)
    # song_bpms_2 = find_bpm(loaded_songs_2)
    # song_bpms_3 = find_bpm(loaded_songs_3)

    feature_extractor(loaded_songs_1)


def directory_loader(directory_path):
    """Function that will retrieve all files from a specified directory path
    note: the directory passed in must have a trailing ' / '
    """
    files = []

    print("Reading Files From.. " + directory_path)
    directory = os.fsencode(directory_path)

    for file in os.listdir(directory):
        filename = str(os.fsdecode(file))
        files.append(directory_path + filename)
    return files


def audio_load(song_paths, type):
    loaded_songs = []
    for song in song_paths:
        try:
            y, sr = librosa.load(song)
            # print("Audio Path: ", song, " Waveform Loaded")
            print("Audio {:10} Waveform Loaded".format(song))
            loaded_songs.append([type, y, sr])

        except:
            print("Failed To Load Audio File")
            print("Closing")

    return loaded_songs


def feature_extractor(songs):
    # initial try with 1 song
    print(songs)


def find_bpm(loaded_audio):
    song_bpms = []
    for song in loaded_audio:
        tempo, beats = librosa.beat.beat_track(song[1], song[2])
        print("Song Type {} is {} BPM".format(song[0], tempo))
        song_bpms.append([song[0], tempo])
    return song_bpms


main()
