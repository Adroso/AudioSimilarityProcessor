from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

AUDIO_PATH_1 = 'Trainer_MusicType1/rock2.mp3'
AUDIO_PATH_2 = 'Trainer_MusicType2/tech1.mp3'
AUDIO_PATH_5 = "TestData/coldplay.mp3"

y, sr = librosa.load(AUDIO_PATH_2, offset=40, duration=10)

# MFCC - Timbre extraction
plt.figure(figsize=(10, 4))
mfccs = librosa.feature.mfcc(y, sr)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title("MFCC Analysis")
plt.tight_layout()

# Spectrograms

D = librosa.stft(y)
D_harmonic, D_percussive = librosa.decompose.hpss(D)
rp = np.max(np.abs(D))
print(D)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
plt.colorbar()
plt.title('Harmonic spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')
plt.tight_layout()

plt.show()
