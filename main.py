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
from sklearn import svm
import numpy as np
from  sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score

# filefinding
# from tkinter.filedialog import askopenfilename


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

    # Loading Each Directory of songs
    loaded_songs_1 = audio_load(training_files_1, 1)
    loaded_songs_2 = audio_load(training_files_2, 2)
    loaded_songs_3 = audio_load(test_files, 3)

    #Calculating each directories songs BPM
    song_bpms_1 = find_bpm(loaded_songs_1)
    song_bpms_2 = find_bpm(loaded_songs_2)
    song_bpms_3 = find_bpm(loaded_songs_3)

    # Calculating Spectrograph for each directory
    song_spectrographs_1 = find_spectrograph(loaded_songs_1)
    song_spectrographs_2 = find_spectrograph(loaded_songs_2)
    song_spectrographs_3 = find_spectrograph(loaded_songs_3)

    # Calaculating MFCC for each directory
    song_mfcc_1 = find_mfcc(loaded_songs_1)
    song_mfcc_2 = find_mfcc(loaded_songs_2)
    song_mfcc_3 = find_mfcc(loaded_songs_3)

    # merging
    training_songs_1 = array_merge(song_bpms_1, song_mfcc_1, song_spectrographs_1)
    training_songs_2 = array_merge(song_bpms_2, song_mfcc_2, song_spectrographs_2)
    test_data = array_merge(song_bpms_3, song_mfcc_3, song_spectrographs_3)


    all_training = training_songs_1 + training_songs_2


    # classifications = audio_classify_svc(training_bpms, song_bpms_3)
    classifications = audio_classify_dtc(all_training, test_data)

    # printing final results
    i = 0
    for song in classifications:
        print(test_files[i], " Is of Music Type: ", song)
        i += 1



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


def find_bpm(loaded_audio):
    song_bpms = []
    for song in loaded_audio:
        tempo, beats = librosa.beat.beat_track(song[1], song[2])
        print("Song Type {} is {} BPM".format(song[0], tempo))
        song_bpms.append([song[0], tempo])
    return song_bpms


def find_spectrograph(loaded_songs):
    print("Calculating Song Spectrograph's")
    # loadedsongs structure [songtype, songwaveform, samplerate]
    song_spectralmaps = []
    for song in loaded_songs:
        D = librosa.stft(song[1])
        # D_harmonic, D_percussive = librosa.decompose.hpss(D)
        # rp = np.max(np.abs(D))
        song_spectralmaps.append([song[0], D])
    return song_spectralmaps


def find_mfcc(loaded_songs):
    print("Calculating Song MFCC")
    song_timbres = []
    for song in loaded_songs:
        mfccs = librosa.feature.mfcc(song[1], song[2])
        song_timbres.append([song[0], mfccs])
    return song_timbres


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


def audio_classify_svc(train_up, test):
    # prepare data for model
    lables = []
    for item in train_up:
        lables.append(item[0])
    train = np.array(train_up)

    #initialise model
    cfl = svm.SVC(kernel='linear', C=1.0)
    cfl.fit(train, lables)
    print("Classified")

    return cfl.predict(test)


def audio_classify_dtc(train_up, test):
    # TODO Fix classifier error with accepting an array as a feature.

    print("Attempting Tree Classification")
    lables = []
    for item in train_up:
        lables.append(item[0])
        print(item)
    train = np.array(train_up)
    #train = train_up

    tree_model = dtc()
    tree_model.fit(train, lables)
    prediction = tree_model.predict(test)
    return prediction


def array_merge(song_bpms, song_mfccs, song_spectro):
    print("Merging Data properties into list")
    song_features = []

    for i in range(1, len(song_bpms)):
        for bpm in song_bpms:
            ind_bpm = bpm[1]
        for mfcc in song_mfccs:
            ind_mfcc = mfcc[1]
        for spectral in song_spectro:
            ind_spectro = spectral[1]

        song_features.append([song_bpms[0], ind_bpm, ind_mfcc, ind_spectro])

    return song_features


main()
