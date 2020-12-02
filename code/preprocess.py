import tensorflow as tf
import os
import scipy.io.wavfile
import numpy as np
import math
from PIL import Image
import time
from pydub import AudioSegment

FFT_LENGTH = 1024
WINDOW_LENGTH = 512
WINDOW_STEP = int(WINDOW_LENGTH / 2)
MAG_MIN = 0
MAG_MAX = 4000000
MAG_RANGE = MAG_MAX - MAG_MIN
PHA_MIN = -math.pi
PHA_MAX = math.pi
PHA_RANGE = PHA_MAX - PHA_MIN
rate = 44100

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

def generate_linear_scale(magnitude_vals, phase_vals):
    """
    Convert the magnitude and phase values to be on a linear scale

    :param magnitude_vals: the magnitude values
    :param phase_vals: the phase values
    :return: the magnitude and phase values on a linear scale
    """

    height = magnitude_vals.shape[0]
    width = magnitude_vals.shape[1]
    for w in range(width):
        for h in range(height):
            magnitude_vals[h,w] = (magnitude_vals[h,w] - MAG_MIN) / (MAG_RANGE) * 255 * 2
            magnitude_vals[h,w] = 188.301 * math.log10(magnitude_vals[h,w] + 1)
            phase_vals[h,w] = (phase_vals[h,w] - PHA_MIN) / (PHA_RANGE) * 255
    return magnitude_vals, phase_vals

def recover_linear_scale(magnitude_vals, phase_vals):
    """
    Convert the linearized magnitude and phase values back to their original values

    :param magnitude_vals: the linearized magnitude values
    :param phase_vals: the linearized phase values
    :return: the original magnitude and phase values
    """

    width = magnitude_vals.shape[1]
    height = magnitude_vals.shape[0]
    for w in range(width):
        for h in range(height):
            phase_vals[h,w] = (phase_vals[h,w] / 255 * PHA_RANGE) + PHA_MIN
            magnitude_vals[h,w] = math.pow(10, magnitude_vals[h,w]/188.301) - 1
            magnitude_vals[h,w] = (magnitude_vals[h,w] / (255*2) * MAG_RANGE) + MAG_MIN
    return magnitude_vals, phase_vals

def calculate_mag_phase_from_signal(signal):
    """
    Calculates the magnitude and phase of the given audio signal

    :param signal: the audio signal of the .wav file
    :return: the magnitude and phase values of the audio signal
    """

    # Create buffer to hold signal and initialize magnitude and phase arrays
    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
    buffer[0:len(signal)] = signal
    height = int(FFT_LENGTH / 2 + 1)
    width = int(len(buffer) / (WINDOW_STEP) - 1)
    magnitude_vals = np.zeros((height, width))
    phase_vals = np.zeros((height, width))

    # Generate magnitude and phase from signal
    for w in range(width):
        buff = np.zeros(FFT_LENGTH)
        step_buff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
        # Apply hanning window
        step_buff = step_buff * np.hanning(WINDOW_LENGTH)
        buff[0:len(step_buff)] = step_buff
        # Buff now contains windowed signal with step length and padded with zeros to the end
        fft = np.fft.rfft(buff)
        for h in range(len(fft)):
            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
            phase = math.atan2(fft[h].imag, fft[h].real)
            magnitude_vals[height-h-1,w] = magnitude
            phase_vals[height-h-1,w] = phase
    
    # Convert magnitude and phase arrays to a linear scale
    magnitude_vals, phase_vals = generate_linear_scale(magnitude_vals, phase_vals)
    return magnitude_vals, phase_vals

def generate_spectrogram_from_mag_phase(file_path_mag, file_path_pha, file_path_out):
    """
    Generates the spectrogram corresponding to a magnitude and phase array

    :param file_path_mag: file path of the magnitude array
    :param file_path_pha: file path of the phase array
    :param file_path_out: directory to save the spectrogram
    :return: None
    """

    # Load in magnitude and phase arrays
    magnitude = np.load(file_path_mag)[0]
    magnitude = np.transpose(magnitude)
    phase = np.load(file_path_pha)[0]
    phase = np.transpose(phase)
    height = magnitude.shape[0]
    width = magnitude.shape[1]

    # Generate a RGB representation of the phase and magnitude arrays
    rgb_array = np.zeros((height, width, 3), 'uint8')
    for w in range(width):
        for h in range(height):
            red = 255 if magnitude[h,w] > 255 else magnitude[h,w]
            green = (magnitude[h,w] - 255) if magnitude[h,w] > 255 else 0
            blue = phase[h,w]
            rgb_array[h,w,0] = int(red)
            rgb_array[h,w,1] = int(green)
            rgb_array[h,w,2] = int(blue)

    # Save generated spectrogram to disk
    spect = Image.fromarray(rgb_array, 'RGB')
    spect.save(file_path_out + "recovered.png")

def recover_signal_from_mag_pha(file_path_mag, file_path_pha, file_path_out):
    """
    Generates the .wav file corresponding to a magnitude and phase array

    :param file_path_mag: file path of the magnitude array
    :param file_path_pha: file path of the phase array
    :param file_path_out: directory to save the .wav
    :return: None
    """

    # Load in magnitude and phase arrays
    magnitude_vals = np.load(file_path_mag)[0]
    magnitude_vals = np.transpose(magnitude_vals)
    phase_vals = np.load(file_path_pha)[0]
    phase_vals = np.transpose(phase_vals)
    magnitude_vals, phase_vals = recover_linear_scale(magnitude_vals, phase_vals)
    width = magnitude_vals.shape[1]
    height = magnitude_vals.shape[0]
    
    # Generate signal from magnitude and phase
    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
    for w in range(width):
        to_inverse = np.zeros(height, dtype=np.complex_)
        for h in range(height):
            magnitude = magnitude_vals[height-h-1,w]
            phase = phase_vals[height-h-1,w]
            to_inverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
        signal = np.fft.irfft(to_inverse)
        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)

    # Save generated signal to disk
    scipy.io.wavfile.write(file_path_out + "recovered.wav", rate, recovered)

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
    arr_mag = None
    arr_phase = None

    for song in audio_clips:
        if (".wav" in song):
            # Grab array position for the song to be inserted
            index = int(song.split()[0])
            # Read in the song
            rate, audData = scipy.io.wavfile.read(file_path_in + song)
            channel1 = audData[:,0]
            # Parse out the first minute of the song
            signal_fragment = channel1[1*rate:60*rate]
            mag, phase = calculate_mag_phase_from_signal(signal_fragment)

            # Reshape data to model specifications and save it in array
            mag = np.transpose(mag)
            phase = np.transpose(phase)
            # Assign magnitude and phase arrays if not created
            if (not initialized):
                initialized = True
                arr_mag = np.zeros((n_songs, mag.shape[0], mag.shape[1]))
                arr_phase = np.zeros((n_songs, phase.shape[0], phase.shape[1]))
            arr_mag[index] = mag
            arr_phase[index] = phase

    np.save(file_path_out + "mashup_mag", arr_mag)
    np.save(file_path_out + "mashup_pha", arr_phase)

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
    arr_mag = None
    arr_phase = None

    for song in audio_clips:
        if (".wav" in song):
            # Grab array position for the song to be inserted
            index = song.split('.')
            first = int(index[0])
            second = int(index[1])
            # Read in the song
            rate, audData = scipy.io.wavfile.read(file_path_in + song)
            channel1 = audData[:,0]
            # Parse out the first minute of the song
            signal_fragment = channel1[1*rate:60*rate]
            mag, phase = calculate_mag_phase_from_signal(signal_fragment)

            # Reshape data to model specifications and save it in array
            mag = np.transpose(mag)
            phase = np.transpose(phase)
            # Assign magnitude and phase arrays if not created
            if (not initialized):
                initialized = True
                arr_mag = np.zeros((n_songs, 2, mag.shape[0], mag.shape[1]))
                arr_phase = np.zeros((n_songs, 2, phase.shape[0], phase.shape[1]))
            arr_mag[first, second] = mag
            arr_phase[first, second] = phase

    # Save arrays to disk
    np.save(file_path_out + "original_mag", arr_mag)
    np.save(file_path_out + "original_pha", arr_phase)

def get_magnitude_data(file_n):
    """
    Pulls the magnitude data from the preprocessed spectrograms.

    :param file_n: The file to pull the data from (in case we have more than 1 file)
    :return: The magnitude of the signal for the training and testing spectrograms, where the
             originals are of size (n_samples, 2, width, height) and the mashes are of size
             (n_samples, width, height)
    """

    train_orig_mag = []
    train_mash_mag = [] 
    test_orig_mag = [] 
    test_mash_mag = []
    return train_orig_mag, train_mash_mag, test_orig_mag, test_mash_mag

def get_phase_data(file_n):
    """
    Pulls the phase data from the preprocessed spectrograms.

    :param file_n: The file to pull the data from (in case we have more than 1 file)
    :return: The phase of the signal for the training and testing spectrograms, where the
             originals are of size (n_samples, 2, width, height) and the mashes are of size
             (n_samples, width, height)
    """

    train_orig_pha = []
    train_mash_pha = [] 
    test_orig_pha = [] 
    test_mash_pha = []
    return train_orig_pha, train_mash_pha, test_orig_pha, test_mash_pha
    
def main():
    #convert_original_to_array("../data/original-wav/", "../data/preprocessed/")
    convert_mashup_to_array("../data/mashup-wav/", "../data/preprocessed/")
    generate_spectrogram_from_mag_phase("../data/preprocessed/mashup_mag.npy", "../data/preprocessed/mashup_pha.npy", "../data/spectrogram/")
    recover_signal_from_mag_pha("../data/preprocessed/mashup_mag.npy", "../data/preprocessed/mashup_pha.npy", "../data/recovered/")


if __name__ == "__main__":
    main()
