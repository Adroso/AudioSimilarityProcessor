"""
Program Created by Adrian Lapico ("Adroso") 2/9/17

Purpose: CP3000 Audio Processing Research Project
Scope: Be able to classify samples of music into genre
Future Implementation: Be able to infer intended emotional response from music to make recommendations
"""

#Imports
import numpy
import matplotlib
import IPython
import librosa




AUDIO_PATH_1 = 'Trainer_MusicType2/tech1.mp3'

def main():
    print("Starting Program..")
    audioLoad()

def audioLoad():
    try:
        y, sr = librosa.load(AUDIO_PATH_1)
        print("Audio Path: ", AUDIO_PATH_1, " Loaded")
    except:
        print("Failed To Load")
main()
