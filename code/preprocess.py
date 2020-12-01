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
    