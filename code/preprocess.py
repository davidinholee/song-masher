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
magnitudeMin = float("inf")
magnitudeMax = float("-inf")
phaseMin = float("inf")
phaseMax = float("-inf")

def convertMp3ToWav(filePath):
    mp3_clips = os.listdir(filePath)
    mp3_clips = audio_clips[mp3_clips]
    for i in mp3_clips:
        sound = AudioSegment.from_mp3(mp3_dir + i)
        sound.export(audio_dir + i + ".wav", format="wav")

def amplifyMagnitudeByLog(d):
    return 188.301 * math.log10(d + 1)

def weakenAmplifiedMagnitude(d):
    return math.pow(10, d/188.301)-1

def generateLinearScale(magnitudePixels, phasePixels, 
                        magnitudeMin, magnitudeMax, phaseMin, phaseMax):
    height = magnitudePixels.shape[0]
    width = magnitudePixels.shape[1]
    magnitudeRange = magnitudeMax - magnitudeMin
    phaseRange = phaseMax - phaseMin
    rgbArray = np.zeros((height, width, 3), 'uint8')
    rgbSingleArray = np.zeros((height, width), 'uint8')
    
    for w in range(width):
        for h in range(height):
            magnitudePixels[h,w] = (magnitudePixels[h,w] - magnitudeMin) / (magnitudeRange) * 255 * 2
            magnitudePixels[h,w] = amplifyMagnitudeByLog(magnitudePixels[h,w])
            phasePixels[h,w] = (phasePixels[h,w] - phaseMin) / (phaseRange) * 255
            red = 255 if magnitudePixels[h,w] > 255 else magnitudePixels[h,w]
            green = (magnitudePixels[h,w] - 255) if magnitudePixels[h,w] > 255 else 0
            blue = phasePixels[h,w]
            rgbArray[h,w,0] = int(red)
            rgbArray[h,w,1] = int(green)
            rgbArray[h,w,2] = int(blue)
            RGBint = (int(red)<<16) + (int(green)<<8) + int(blue)
            rgbSingleArray[h,w] = RGBint
    rgbSingleArray = np.transpose(rgbSingleArray)
    return rgbArray, rgbSingleArray, magnitudePixels, phasePixels

def recoverLinearScale(rgbArray, magnitudeMin, magnitudeMax, 
                       phaseMin, phaseMax):
    width = rgbArray.shape[1]
    height = rgbArray.shape[0]
    magnitudeVals = rgbArray[:,:,0].astype(float) + rgbArray[:,:,1].astype(float)
    phaseVals = rgbArray[:,:,2].astype(float)
    phaseRange = phaseMax - phaseMin
    magnitudeRange = magnitudeMax - magnitudeMin
    for w in range(width):
        for h in range(height):
            phaseVals[h,w] = (phaseVals[h,w] / 255 * phaseRange) + phaseMin
            magnitudeVals[h,w] = weakenAmplifiedMagnitude(magnitudeVals[h,w])
            magnitudeVals[h,w] = (magnitudeVals[h,w] / (255*2) * magnitudeRange) + magnitudeMin
    return magnitudeVals, phaseVals


def generateSpectrogramForWave(signal):
    start_time = time.time()
    global magnitudeMin
    global magnitudeMax
    global phaseMin
    global phaseMax
    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
    buffer[0:len(signal)] = signal
    height = int(FFT_LENGTH / 2 + 1)
    width = int(len(buffer) / (WINDOW_STEP) - 1)
    magnitudePixels = np.zeros((height, width))
    phasePixels = np.zeros((height, width))

    for w in range(width):
        buff = np.zeros(FFT_LENGTH)
        stepBuff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
        # apply hanning window
        stepBuff = stepBuff * np.hanning(WINDOW_LENGTH)
        buff[0:len(stepBuff)] = stepBuff
        #buff now contains windowed signal with step length and padded with zeroes to the end
        fft = np.fft.rfft(buff)
        for h in range(len(fft)):
            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
            if magnitude > magnitudeMax:
                magnitudeMax = magnitude 
            if magnitude < magnitudeMin:
                magnitudeMin = magnitude 

            phase = math.atan2(fft[h].imag, fft[h].real)
            if phase > phaseMax:
                phaseMax = phase
            if phase < phaseMin:
                phaseMin = phase
            magnitudePixels[height-h-1,w] = magnitude
            phasePixels[height-h-1,w] = phase
    rgbArray, rgbSingleArray, mag, phase = generateLinearScale(magnitudePixels, phasePixels,
                                  magnitudeMin, magnitudeMax, phaseMin, phaseMax)
    elapsed_time = time.time() - start_time
    print('%.2f' % elapsed_time, 's', sep='')
    img = Image.fromarray(rgbArray, 'RGB')
    return img, rgbArray, rgbSingleArray, mag, phase

def recoverSignalFromSpectrogram(filePath, name):
    img = Image.open(filePath)
    data = np.array( img, dtype='uint8' )
    width = data.shape[1]
    height = data.shape[0]

    magnitudeVals, phaseVals \
    = recoverLinearScale(data, magnitudeMin, magnitudeMax, phaseMin, phaseMax)
    
    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
    for w in range(width):
        toInverse = np.zeros(height, dtype=np.complex_)
        for h in range(height):
            magnitude = magnitudeVals[height-h-1,w]
            phase = phaseVals[height-h-1,w]
            toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
        signal = np.fft.irfft(toInverse)
        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)
    
    scipy.io.wavfile.write(rec_dir + name + "Recovered.wav", rate, recovered)
# recoverSignalFromSpectrogram(spect_dir + "spectrogram.png")

# Converts all wav files in a diretory to a single array
def convertToMashupArray(filePath):
    audio_clips = os.listdir(filePath)
    audio_clips = audio_clips[1:]
    l = len(audio_clips)
    print(audio_clips)
    arr_mag = np.zeros((l, 20499, 513))
    arr_phase = np.zeros((l, 20499, 513))
    for c, i in enumerate(audio_clips):
        index = i.split()
        fst = int(index[0])
        rate, audData = scipy.io.wavfile.read(filePath + i)
        channel1 = audData[:,0]
        channel2 = audData[:,1]
        # This should be the first 2 minutes of the song
        signal_fragment = channel1[1*rate:120*rate]

        img, rgbArray, rgbSingleArray, mag, phase = generateSpectrogramForWave(signal_fragment)
        mag = np.transpose(mag)
        phase = np.transpose(phase)
        arr_mag[fst] = mag
        arr_phase[fst] = phase

    print(arr_mag.shape)
    np.save(arrays_dir + "mashup_mag", arr_mag)
    np.save(arrays_dir + "mashup_phase", arr_phase)

def convertOriginalToArray(filePath):
    audio_clips = os.listdir(filePath)
    # audio_clips = audio_clips[1:]
    print(audio_clips)
    l = int(len(audio_clips)/2)
    # 4995 for 30 seconds
    arr_mag = np.zeros((l, 2, 20499, 513))
    arr_phase = np.zeros((l, 2, 20499, 513))
    for i in audio_clips:
        index = i.split('.')
        fst = int(index[0])
        snd = int(index[1])
        rate, audData = scipy.io.wavfile.read(filePath + i)
        channel1 = audData[:,0]
        channel2 = audData[:,1]
        # This should be the first 2 minutes of the song
        signal_fragment = channel1[1*rate:120*rate]
        img, rgbArray, rgbSingleArray, mag, phase = generateSpectrogramForWave(signal_fragment)
        mag = np.transpose(mag)
        phase = np.transpose(phase)
        arr_mag[fst, snd] = mag
        arr_phase[fst, snd] = phase

    print(arr_mag.shape)
    np.save(arrays_dir + "original_mag", arr_mag)
    np.save(arrays_dir + "original_phase", arr_phase)

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
    
if __name__ == "__main__":
    audio_dir = "../audio/"
    ori_dir = "../original/"
    spect_dir = "../spect/"
    arrays_dir = "../arrays/"
    rec_dir = "../recovered/"
    mp3_dir = "../mp3/"

    convertOriginalToArray(ori_dir)
    convertToMashupArray(audio_dir)