"""
Program Created by Adrian Lapico ("Adroso") 25/9/17

Purpose: CP3000 Audio Processing Research Project - Second Attempt Starting From Scratch
Scope: Be able to classify samples of music into genre
Future Implementation: Be able to infer intended emotional response from music to make recommendations
"""

import matplotlib.style as ms
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import librosa
import librosa.display
import os
import numpy as np


def main():
    # note position 0 in songs array corresponds to position 2 in songtypes
    SONGS = ['Trainer_MusicType1/rock5.mp3', 'Trainer_MusicType1/rock4.mp3', 'Trainer_MusicType2/tech3.mp3',
             'Trainer_MusicType2/tech5.mp3']
    SONGTYPES = ['rock', 'rock', 'techno', 'techno']

    TESTSONGS = []
    TESTSONGSTYPES = []

    raw_waveforms = audio_load(SONGS)
    raw_features = extract_features(raw_waveforms)
    audio_classifyer(raw_features, SONGTYPES)


def audio_load(song_paths):
    """Loads via librosa all song within a passed in list of song paths"""
    loaded_songs = []
    for song in song_paths:
        try:
            waveForms, sampleRate = librosa.load(song)
            print("Audio {:10} Waveform Loaded".format(song))
            loaded_songs.append([waveForms, sampleRate])
            #loaded_songs.append(y)
        except:
            print("Failed To Load Audio File")
            print("Closing")

    return loaded_songs


def extract_features(raw_sounds):
    # print(raw_sounds[0][1], genres)

    audioFeatures = []

    for i in raw_sounds:
        # print(i[0], i[1])
        print("Processing Audio Features")
        raw_mfccs = np.mean(librosa.feature.mfcc(y=i[0], sr=i[1]).T, axis=0)
        # averaging whole array
        mfccs = np.average(raw_mfccs)

        # calculating bpm
        bpm, beats = librosa.beat.beat_track(i[0], i[1])

        audioFeatures.append([mfccs, bpm])

    return audioFeatures


def audio_classifyer(features, lables):
    classifyer_data = np.asanyarray(features)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(classifyer_data)



    # print(averaged_mfccs)
    # print(lables)





main()
