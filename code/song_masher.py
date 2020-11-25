import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import SongMasher
import sys
import random

def train(model, train_originals, train_mashes):
    """
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_originals: training data, list of the original two songs that were used to create the corresponding mash 
           of shape (num_examples, 2, n_slices, spectrogram_height, slice_width, n_channels)
	:param train_mashes: training labels, list of the mashed songs derived from the corresponding two training data songs
           of shape (num_examples, n_slices, spectrogram_height, slice_width, n_channels)
	:return: None
	"""

    # Shuffle inputs
    idx = np.arange(train_originals.shape[0])
    idx = tf.random.shuffle(idx)
    train_originals = tf.gather(train_originals, idx)
    train_mashes = tf.gather(train_mashes, idx)
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    # Iterate through each batch
    for i in range(0, train_originals.shape[0], model.batch_size):
        # Get batch of data
        orig_batch = train_originals[i:i+model.batch_size]
        mash_batch = train_mashes[i:i+model.batch_size]

        # Calculate predictions and loss
        with tf.GradientTape() as tape:
            artif_mashes = model(orig_batch)
            loss = model.loss_function(artif_mashes, mash_batch)

        # Apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def main():
    # TODO: Use the model to train a song mashing network
	print("Running preprocessing...")
	train_originals, train_mashes, test_originals, test_mashes = get_data()
	print("Preprocessing complete.")

	model = SongMasher(*(train_originals.shape[1])) 
	
	# Train and test Model for 1 epoch.
	train(model, train_originals, train_mashes)
	cross_corr = 0
	print("Loss: %.2d", cross_corr)


if __name__ == '__main__':
    main()
