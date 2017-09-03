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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np

# filefinding
from tkinter.filedialog import askopenfilename

AUDIO_PATH_1 = 'Trainer_MusicType1/rock2.mp3'
AUDIO_PATH_2 = 'Trainer_MusicType2/tech1.mp3'
AUDIO_PATH_5 = "TestData/coldplay.mp3"

TRAINING_1 = 'Trainer_MusicType1/'
TRAINING_2 = 'Trainer_MusicType2/'
TEST = "TestData/"


def main():
    print("Starting Program..")

    training_files_1 = directory_loader(TRAINING_1)
    training_files_2 = directory_loader(TRAINING_2)
    test_files = directory_loader(TEST)

    # if want to find audio on system
    # filename = askopenfilename()
    # y, sr = audioLoad(filename)
    loaded_songs_1 = audio_load(training_files_1, 1)
    loaded_songs_2 = audio_load(training_files_2, 2)
    loaded_songs_3 = audio_load(test_files, 3)

    song_bpms_1 = find_bpm(loaded_songs_1)
    song_bpms_2 = find_bpm(loaded_songs_2)
    song_bpms_3 = find_bpm(loaded_songs_3)

    # merging directories for training data
    training_bpms = song_bpms_1 + song_bpms_2

    audio_classify(training_bpms, song_bpms_3)



def audio_load(song_paths, type):
    # TODO Look into creating a class to manage songs for implementation
    loaded_songs = []
    for song in song_paths:
        try:
            y, sr = librosa.load(song)
            print("Audio Path: ", song, " Loaded into Analyzer")
            loaded_songs.append([type, y, sr])

        except:
            print("Failed To Load Audio File")
            print("Closing")

    return loaded_songs


def find_bpm(loaded_audio):
    song_bpms = []
    for song in loaded_audio:
        tempo, beats = librosa.beat.beat_track(song[1], song[2])
        print("Song Type {} is {} BPM".format(song[0], tempo))
        song_bpms.append([song[0], tempo])
    return song_bpms


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


def audio_classify(train_up, test):
    lables = []
    for item in train_up:
        lables.append(item[0])

    train = np.array(train_up)

    cfl = svm.SVC(kernel='linear', C=1.0)

    cfl.fit(train, lables)
    print(cfl.predict(test))


main()
