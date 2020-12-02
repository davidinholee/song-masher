from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import librosa

samplerate, data = wavfile.read("../data/a.wav.wav")
print(samplerate)
print(data.shape)
samplerate, data = librosa.load('../data/a.wav.wav')
print(samplerate.shape)
print(data)