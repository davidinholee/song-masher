import os
import numpy as np
import math
from pydub import AudioSegment
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

SAMPLE_RATE = 8000

def convert_mp3_to_wav(file_path_in, file_path_out):
    """
    Converts the mp3 files in the directory to wav files

    :param file_path_in: directory containing the mp3 files
    :param file_path_out: directory to store the wav files
    :return: None
    """

    mp3_clips = os.listdir(file_path_in)
    for song in mp3_clips:
        sound = AudioSegment.from_mp3(file_path_in + song)
        sound.export(file_path_out + song + ".wav", format="wav")

def generate_spectrogram(file_path_in, file_path_out, plot_title):
    """
    Generates the spectrogram corresponding to a signal array

    :param file_path_in: file path of the signal array
    :param file_path_out: file path of the to be generated spectrogram
    :param plot_title: title of the spectrogram
    :return: None
    """

    # Load in signal array
    aud_data = np.load(file_path_in)[0]
    # Generate spectrogram data and scale it to represent decibels
    spect_raw = librosa.feature.melspectrogram(y=aud_data, sr=SAMPLE_RATE)
    spect_dB = librosa.power_to_db(spect_raw, ref=np.max)
    # Format spectrogram data into an image using matplotlib
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    img = librosa.display.specshow(spect_dB, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=plot_title)
    fig.savefig(file_path_out)

def generate_audio(file_path_in, file_path_out):
    """
    Generates the wav file corresponding to a signal array

    :param file_path_in: file path of the signal array
    :param file_path_out: file path of the to be generated wav
    :return: None
    """

    # Load in signal array
    aud_data = np.load(file_path_in)[0]
    # Use soundfile to write signal to a wav file
    sf.write(file_path_out, aud_data, SAMPLE_RATE)

def convert_mashup_to_array(file_path_in, file_path_out):
    """
    Converts all wav files in the mashup diretory to a single array

    :param file_path_in: directory of all the mashups
    :param file_path_out: directory to save the array
    :return: None
    """

    # Get list of all .wav files in directory
    audio_clips = os.listdir(file_path_in)
    n_songs = len(audio_clips) - 1
    # Initialize the magnitude and phase arrays
    initialized = False
    signal = None
    for song in audio_clips:
        if (".wav" in song):
            # Grab array position for the song to be inserted
            index = int(song.split()[0])
            # Read in the song
            aud_data, _ = librosa.load(file_path_in + song, sr=SAMPLE_RATE)
            # Parse out the first minute of the song
            signal_fragment = aud_data[:60*SAMPLE_RATE]
            # Assign signal array if not created
            if (not initialized):
                initialized = True
                signal = np.zeros((n_songs, signal_fragment.shape[0]), dtype=np.float32)
            signal[index] = signal_fragment
            signal[index] = signal_fragment

    # Save array to disk
    np.save(file_path_out + "mashup", signal)

def convert_original_to_array(file_path_in, file_path_out):
    """
    Converts all wav files in the original song diretory to a single array

    :param file_path_in: directory of all the original songs
    :param file_path_out: directory to save the array
    :return: None
    """

    # Get list of all .wav files in directory
    audio_clips = os.listdir(file_path_in)
    n_songs = int((len(audio_clips) - 1)/2)
    # Initialize the magnitude and phase arrays
    initialized = False
    signal = None
    for song in audio_clips:
        if (".wav" in song):
            # Grab array position for the song to be inserted
            index = song.split('.')
            first = int(index[0])
            second = int(index[1])
            # Read in the song
            aud_data, _ = librosa.load(file_path_in + song, sr=SAMPLE_RATE)
            # Parse out the first minute of the song
            signal_fragment = aud_data[:60*SAMPLE_RATE]
            # Assign signal array if not created
            if (not initialized):
                initialized = True
                signal = np.zeros((n_songs, 2, signal_fragment.shape[0]), dtype=np.float32)
            signal[first, second] = signal_fragment

    # Save array to disk
    np.save(file_path_out + "original", signal)

def get_data(file_path_orig, file_path_mash, split):
    """
    Pulls the signal data from the preprocessed audio files.

    :param file_path_orig: The file that contains the original songs signal array
    :param file_path_mash: The file that contains the mashed songs signal array
    :return: The signal arrays for the training and testing original/mashed songs, where 
             the originals are of size (n_samples, 2, n_timesteps) and the mashes are of 
             size (n_samples, width, n_timesteps)
    """

    originals = np.load(file_path_orig)
    mashes = np.load(file_path_mash)
    n_samples = originals.shape[0]
    train_orig = originals[:int(n_samples*split)]
    train_mash = mashes[:int(n_samples*split)]
    test_orig = originals[int(n_samples*split):]
    test_mash = mashes[int(n_samples*split):]
    return train_orig, train_mash, test_orig, test_mash
    
def main():
    convert_original_to_array("../data/original-wav/", "../data/preprocessed/")
    convert_mashup_to_array("../data/mashup-wav/", "../data/preprocessed/")
    generate_spectrogram("../data/preprocessed/mashup.npy", "../data/spectrogram/test.png", "Test Spectrogram")
    generate_audio("../data/preprocessed/mashup.npy", "../data/audio/test.wav")


if __name__ == "__main__":
    main()
