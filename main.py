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
    loaded_songs = audio_load(training_files_1)
    # find_bpm()


def audio_load(song_paths):
    # TODO Load every audio file from a specified directory and save the outputs
    loaded_songs = []
    for song in song_paths:
        try:
            y, sr = librosa.load(song)
            print("Audio Path: ", song, " Loaded into Analyzer")
            loaded_songs.append([song, y, sr])

        except:
            print("Failed To Load Audio File")
            print("Closing")
    return loaded_songs


def find_bpm(y, sr, audio_paths):
    for song in audio_paths:
        tempo, beats = librosa.beat.beat_track(y, sr)
        print("{} is {} BPM".format(audio_paths.split('/', 1)[-1], tempo))


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

def audio_classify():
    pass


main()
