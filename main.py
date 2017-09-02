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

    y, sr = audio_load(AUDIO_PATH_1)
    find_bpm(y, sr, AUDIO_PATH_1)


def audio_load(audio_path):
    # TODO Load every audio file from a specified directory and save the outputs
    try:
        print("Loading..... ", audio_path)
        y, sr = librosa.load(audio_path)
        print("Audio Path: ", audio_path, " Loaded")
        return y, sr
    except:
        print("Failed To Load Audio File")
        print("Closing")


def find_bpm(y, sr, audio_path):
    tempo, beats = librosa.beat.beat_track(y, sr)
    print("{} is {} BPM".format(audio_path.split('/', 1)[-1], tempo))


def directory_loader(directory_path):
    """Function that will retrieve all files from a specified directory path
    note: the directory passed in must have a trailing ' / '
    """
    files = []
    directory = os.fsencode(directory_path)

    for file in os.listdir(directory):
        filename = str(os.fsdecode(file))
        files.append(directory_path + filename)
    print(files)
    return files

def audio_classify():
    pass


main()
