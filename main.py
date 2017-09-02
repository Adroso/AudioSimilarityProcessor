"""
Program Created by Adrian Lapico ("Adroso") 2/9/17

Purpose: CP3000 Audio Processing Research Project
Scope: Be able to classify samples of music into genre
Future Implementation: Be able to infer intended emotional response from music to make recommendations
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import IPython.display
import librosa
import librosa.display

#filefinding
from tkinter.filedialog import askopenfilename


AUDIO_PATH_1 = 'Trainer_MusicType1/rock2.mp3'
AUDIO_PATH_2 = 'Trainer_MusicType2/tech1.mp3'
AUDIO_PATH_5 ="TestData/coldplay.mp3"

def main():
    print("Starting Program..")

    #if want to find audio on system
    #filename = askopenfilename()
    #y, sr = audioLoad(filename)

    y, sr = audioLoad(AUDIO_PATH_1)
    findBPM(y, sr, AUDIO_PATH_1)


def audioLoad(audioPath):
    #TODO Load every audio file from a specified directory and save the outputs
    try:
        print("Loading..... ", audioPath)
        y, sr = librosa.load(audioPath)
        print("Audio Path: ", audioPath, " Loaded")
        return y, sr
    except:
        print("Failed To Load Audio File")
        print("Closing")

def findBPM(y, sr, audioPath):
    tempo, beats = librosa.beat.beat_track(y, sr)
    print("{} is {} BPM".format(audioPath.split('/', 1)[-1], tempo))

def audioClassify():
    pass
main()
