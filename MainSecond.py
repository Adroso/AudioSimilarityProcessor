"""
Program Created by Adrian Lapico ("Adroso") 25/9/17

Purpose: CP3000 Audio Processing Research Project - Second Attempt Starting From Scratch
Scope: Be able to classify samples of music into genre to make recommendations
Future Implementation: Be able to infer intended emotional response from music
"""


from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import tree
import librosa
import librosa.display
import numpy as np

# note position 0 in songs array corresponds to position 2 in songtypes
SONGS = ['Trainer_MusicType1/rock5.mp3', 'Trainer_MusicType1/rock4.mp3', 'Trainer_MusicType1/rock3.mp3',
         'Trainer_MusicType1/rock2.mp3', 'Trainer_MusicType2/tech3.mp3',
         'Trainer_MusicType2/tech5.mp3', 'Trainer_MusicType2/tech1.mp3', 'Trainer_MusicType2/tech4.mp3']
SONGTYPES = [0, 0, 0, 0, 1, 1, 1, 1]
# SONGTYPES = ['rock', 'rock', 'techno', 'techno']

TESTSONGS = ['Trainer_MusicType3/techLuke.mp3', 'Trainer_MusicType3/technoAdri.mp3', 'Trainer_MusicType3/rockLuke2.m4a',
             'Trainer_MusicType3/unknownLuke.mp3', 'Trainer_MusicType3/unknownLuke.mp3',
             'Trainer_MusicType3/orcestral_type3.mp3', 'Trainer_MusicType3/technoJap.mp3',
             'Trainer_MusicType3/The Chainsmokers - Dont Let Me Down (Illenium Remix).mp3']


# TESTSONGS = ['Trainer_MusicType1/rock3.mp3', 'Trainer_MusicType1/rock5.mp3', 'Trainer_MusicType2/tech4.mp3',
#              'Trainer_MusicType2/tech2.mp3', 'Trainer_MusicType3/orcestral_type3.mp3']

# TESTSONGSTYPES = ['rock', 'techno']

def main():

    raw_waveforms = audio_load(SONGS)
    test_waveforms = audio_load(TESTSONGS)
    raw_features = extract_features(raw_waveforms)
    test_features = extract_features(test_waveforms)

    audio_classifyer(raw_features, SONGTYPES, test_features)
    decision_tree_classifyer(raw_features, SONGTYPES, test_features)
    # mini_batch(raw_features, SONGTYPES, test_features)


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
        print("Processing Audio Features {}".format(i))
        wavform = i[0]
        samrate = i[1]

        # MFCC  Mel-frequency cepstral coefficients
        raw_mfccs = np.mean(librosa.feature.mfcc(y=wavform, sr=samrate).T, axis=0)
        mfccs = np.average(raw_mfccs)

        # BPM
        bpm, beats = librosa.beat.beat_track(wavform, samrate)

        # STFT (fourer Transform)
        stft = np.array(librosa.stft(wavform))

        # CHROMA     Compute a chromagram from a waveform or power spectrogram.
        raw_chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=samrate).T, axis=0)
        chroma = np.average(raw_chroma)

        # MEL   Compute a mel-scaled spectrogram.
        raw_mel = np.mean(librosa.feature.melspectrogram(wavform, sr=samrate).T, axis=0)
        mel = np.average(raw_mel)

        # CONTRAST   Compute spectral contrast
        raw_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=samrate).T, axis=0)
        contrast = np.average(raw_contrast)

        # TONNETZ   Computes the tonal centroid features
        raw_tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(wavform),
                                                      sr=samrate).T, axis=0)
        tonnetz = np.average(raw_tonnetz)

        print("chroma  ", chroma, "  | mel  ", mel, " | Contrast  ", contrast, " | tonnetz  ", tonnetz)

        audioFeatures.append([mfccs, bpm, chroma, mel, contrast, tonnetz])


    return audioFeatures


def audio_classifyer(features, lables, testingFeatures):
    print("KMEANS CLUSTERING")
    classifyer_data = np.asanyarray(features)
    classifyer_lables = np.asanyarray(lables)
    testing_data = np.asanyarray(testingFeatures)

    # print(classifyer_data)
    # print(classifyer_lables)
    #print(testing_data)

    kmeans = KMeans(n_clusters=2, random_state=1).fit(classifyer_data)
    # kmeans.labels_ = classifyer_lables

    results = kmeans.predict(testing_data)

    print("KMEANS")
    song = 0
    for i in results:
        if i == 1:
            txtLab = "Techno/Electronic"
        else:
            txtLab = "Rock"

        print("{} song has been clustered into song: type {}.".format(TESTSONGS[song], txtLab))
        song += 1


def decision_tree_classifyer(features, lables, testingFeatures):
    print("DECISION TREE CLASSIFYER")
    classifyer_data = np.asanyarray(features)
    classifyer_lables = np.asanyarray(lables)
    testing_data = np.asanyarray(testingFeatures)

    dt_clf = tree.DecisionTreeClassifier().fit(classifyer_data, classifyer_lables)
    dt_clf.predict(testing_data)






def mini_batch(features, lables, testingFeatures):
    classifyer_data = np.asanyarray(features)
    classifyer_lables = np.asanyarray(lables)
    testing_data = np.asanyarray(testingFeatures)

    miniK = MiniBatchKMeans(n_clusters=2, batch_size=50,
                            n_init=2, max_no_improvement=10, verbose=0).fit(classifyer_data)
    # kmeans.labels_ = classifyer_lables

    results = miniK.predict(testing_data)

    print("MINI BTACH KMEANS")
    song = 0
    for i in results:
        if i == 1:
            txtLab = "Techno/Electronic"
        else:
            txtLab = "Rock"

        print("{} song has been clustered into song: type {}.".format(TESTSONGS[song], txtLab))
        song += 1




main()
