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
           of shape (num_examples, 2, spectrogram_width, spectrogram_height)
	:param train_mashes: training labels, list of the mashed songs derived from the corresponding two training data songs
           of shape (num_examples, spectrogram_width, spectrogram_height)
	:return: None
	"""

    # Shuffle inputs
    idx = np.arange(train_originals.shape[0])
    idx = tf.random.shuffle(idx)
    train_originals = tf.gather(train_originals, idx)
    train_originals1 = train_originals[:,0,:,:]
    train_originals2 = train_originals[:,1,:,:]
    train_mashes = tf.gather(train_mashes, idx)
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    # Iterate through each batch
    for i in range(0, train_originals.shape[0], model.batch_size):
        # Get batch of data
        orig_batch1 = tf.cast(train_originals1[i:i+model.batch_size], np.float32)
        orig_batch2 = tf.cast(train_originals2[i:i+model.batch_size], np.float32)
        mash_batch = tf.cast(train_mashes[i:i+model.batch_size], np.float32)

        # Calculate predictions and loss
        with tf.GradientTape() as tape:
            artif_mashes = model(orig_batch1, orig_batch2)
            artif_mashes = tf.reshape(artif_mashes, [model.batch_size, artif_mashes.shape[1] * artif_mashes.shape[2]])
            mash_batch = tf.reshape(mash_batch, [model.batch_size, mash_batch.shape[1] * mash_batch.shape[2]])
            loss = model.loss_function(artif_mashes, mash_batch)

        # Apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_originals, test_mashes):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_originals: testing data, list of the original two songs that were used to create the corresponding mash 
            of shape (num_examples, 2, spectrogram_width, spectrogram_height)
    :param test_mashes: testing labels, list of the mashed songs derived from the corresponding two training data songs
            of shape (num_examples, spectrogram_width, spectrogram_height)
    :return: Average batch loss of the model on the testing set
    """

    # Shuffle inputs
    idx = np.arange(test_originals.shape[0])
    idx = tf.random.shuffle(idx)
    test_originals = tf.gather(test_originals, idx)
    test_originals1 = test_originals[:,0,:,:]
    test_originals2 = test_originals[:,1,:,:]
    test_mashes = tf.gather(test_mashes, idx)

    losses = []
    # Iterate through each batch
    for i in range(0, test_originals.shape[0], model.batch_size):
        # Get batch of data
        orig_batch1 = tf.cast(test_originals1[i:i+model.batch_size], np.float32)
        orig_batch2 = tf.cast(test_originals2[i:i+model.batch_size], np.float32)
        mash_batch = tf.cast(test_mashes[i:i+model.batch_size], np.float32)

        # Calculate predictions and loss
        artif_mashes = model(orig_batch1, orig_batch2)
        artif_mashes = tf.reshape(artif_mashes, [model.batch_size, artif_mashes.shape[1] * artif_mashes.shape[2]])
        mash_batch = tf.reshape(mash_batch, [model.batch_size, mash_batch.shape[1] * mash_batch.shape[2]])
        losses.append(model.loss_function(artif_mashes, mash_batch))
    return np.average(losses)

def main():
    print("Running preprocessing...")
    # Gather preprocessed training and testing data
    train_originals, train_mashes, test_originals, test_mashes = get_data()
    # Convert RGB values to a 1-D scalar
    train_originals = convert_to_rgb(train_originals)
    train_mashes = convert_to_rgb(train_mashes)
    test_originals = convert_to_rgb(train_originals)
    test_mashes = convert_to_rgb(train_mashes)
    print("Preprocessing complete.")

    train_originals = np.random.normal(0, 0.1, (4, 2, 50, 20))
    train_mashes = np.random.normal(0, 0.1, (4, 50, 20))
    test_originals = np.random.normal(0, 0.1, (4, 2, 50, 20))
    test_mashes = np.random.normal(0, 0.1, (4, 50, 20))

    # Train and test model for 1 epoch.
    for epoch in range(5):
        model = SongMasher(train_originals.shape[2], train_originals.shape[3]) 
        train(model, train_originals, train_mashes)
        loss = test(model, test_originals, test_mashes)
        print("Epoch %d Test Loss: %.3f" % (epoch, loss), flush=True)


if __name__ == '__main__':
    main()
