import os
import numpy as np
import math
from pydub import AudioSegment
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

SAMPLE_RATE = 8000
WINDOW_LENGTH = 512
NFFT = 512
MAG_MAX = 80

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
    aud_data = np.load(file_path_in)[:,0]
    # Denormalize signal magnitudes
    aud_data[0] = aud_data[0] * 60
    if (len(aud_data.shape) == 3):
        # Mashup songs
        magnitude = np.transpose(aud_data[0])
        phase = np.transpose(aud_data[1])
        # Calculate original audio signal values
        transformed = magnitude * phase
        signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
        # Retransform to get sparser data
        transformed = np.abs(librosa.stft(signal))
        # Format spectrogram data into an image using matplotlib
        fig, ax = plt.subplots()
        fig.set_size_inches(30, 5)
        img = librosa.display.specshow(librosa.amplitude_to_db(transformed, ref=np.max), y_axis='log', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title=plot_title)
        fig.savefig(file_path_out + ".png")
    else:
        # Original songs
        for i in range(2):
            magnitude = np.transpose(aud_data[0,i])
            phase = np.transpose(aud_data[1,i])
            # Calculate original audio signal values
            transformed = magnitude * phase
            signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
            # Retransform to get sparser data
            transformed = np.abs(librosa.stft(signal))
            # Format spectrogram data into an image using matplotlib
            fig, ax = plt.subplots()
            fig.set_size_inches(30, 5)
            img = librosa.display.specshow(librosa.amplitude_to_db(transformed, ref=np.max), y_axis='log', x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title=plot_title+" "+str(i+1))
            fig.savefig(file_path_out + "_" + str(i+1) + ".png")

def generate_audio(file_path_in, file_path_out):
    """
    Generates the wav file corresponding to the given signal array

    :param file_path_in: file path of the signal array
    :param file_path_out: file path of the to be generated wav
    :return: None
    """

    # Load in signal array
    aud_data = np.load(file_path_in)[:,0]
    # Denormalize signal magnitudes
    aud_data[0] = aud_data[0] * 60
    if (len(aud_data.shape) == 3):
        # Mashup song
        magnitude = np.transpose(aud_data[0])
        phase = np.transpose(aud_data[1])
        # Calculate original audio signal values
        transformed = magnitude * phase
        signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
        # Use soundfile to write signal to a wav file
        sf.write(file_path_out + ".wav", signal, SAMPLE_RATE)
    else:
        # Original songs
        for i in range(2):
            magnitude = np.transpose(aud_data[0,i])
            phase = np.transpose(aud_data[1,i])
            # Calculate original audio signal values
            transformed = magnitude * phase
            signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
            # Use soundfile to write signal to a wav file
            sf.write(file_path_out + "_" + str(i+1) + ".wav", signal, SAMPLE_RATE)

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
            aud_data = aud_data[:60*SAMPLE_RATE]
            # Calculate the magnitude and phase arrays
            transformed = librosa.stft(aud_data, win_length=WINDOW_LENGTH, n_fft=NFFT)
            magnitude, phase = librosa.magphase(transformed)
            # Assign signal array if not created
            if (not initialized):
                initialized = True
                signal = np.zeros((2, n_songs, magnitude.shape[1], magnitude.shape[0]), dtype=np.float32)
            signal[0,index] = np.transpose(magnitude)
            signal[1,index] = np.transpose(phase)

    # Normalize array so that magnitude values are between 0 and 1 (phase values are already normalized)
    signal[0] = signal[0] / MAG_MAX
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
            # Read in the song and calculate the magnitude and phase arrays
            aud_data, _ = librosa.load(file_path_in + song, sr=SAMPLE_RATE)
            # Parse out the first minute of the song
            aud_data = aud_data[:60*SAMPLE_RATE]
            # Calculate the magnitude and phase arrays
            transformed = librosa.stft(aud_data, win_length=WINDOW_LENGTH, n_fft=NFFT)
            magnitude, phase = librosa.magphase(transformed)
            # Assign signal array if not created
            if (not initialized):
                initialized = True
                signal = np.zeros((2, n_songs, 2, magnitude.shape[1], magnitude.shape[0]), dtype=np.float32)
            signal[0, first, second] = np.transpose(magnitude)
            signal[1, first, second] = np.transpose(phase)

    # Normalize array so that magnitude values are between 0 and 1 (phase values are already normalized)
    signal[0] = signal[0] / MAG_MAX
    # Save array to disk
    np.save(file_path_out + "original", signal)

def get_data(file_path_orig, file_path_mash, split):
    """
    Pulls the signal data from the preprocessed audio files.

    :param file_path_orig: The file that contains the original songs signal array
    :param file_path_mash: The file that contains the mashed songs signal array
    :return: The magnitude and phase arrays that make up the audio signal for each 
             training and testing original/mashed song, where the original arrays 
             are of size (n_samples, 2, n_timesteps) and the mashed arrays are of 
             size (n_samples, width, n_timesteps).
    """

    # Read in magnitude and phase arrays for both original and mashed songs
    originals = np.load(file_path_orig)
    orig_mag = originals[0]
    orig_pha = originals[1]
    mashes = np.load(file_path_mash)
    mash_mag = mashes[0]
    mash_pha = mashes[1]
    split_index = int(orig_mag.shape[0]*split)

    # Split the magnitude and phase arrays between the training and testing sets for both original and mashed songs
    train_orig_mag = orig_mag[:split_index]
    train_orig_pha = orig_pha[:split_index]
    train_mash_mag = mash_mag[:split_index]
    train_mash_pha = mash_pha[:split_index]
    test_orig_mag = orig_mag[split_index:]
    test_orig_pha = orig_pha[split_index:]
    test_mash_mag = mash_mag[split_index:]
    test_mash_pha = mash_pha[split_index:]
    return train_orig_mag, train_orig_pha, train_mash_mag, train_mash_pha, test_orig_mag, test_orig_pha, test_mash_mag, test_mash_pha
    
def prep():
    convert_original_to_array("../data/original-wav/", "../data/preprocessed/")
    convert_mashup_to_array("../data/mashup-wav/", "../data/preprocessed/")
    generate_spectrogram("../data/preprocessed/mashup.npy", "../data/spectrogram/test", "Test Spectrogram")
    generate_audio("../data/preprocessed/mashup.npy", "../data/audio/test")


if __name__ == "__main__":
    prep()
