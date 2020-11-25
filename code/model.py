import numpy as np
import tensorflow as tf
import model_funcs as transformer

# TODO: Define the model architecture here
class SongMasher(tf.keras.Model):
    def __init__(self):
        super(SongMasher, self).__init__()

        # Define batch size and optimizer/learning rate
        self.batch_size = 2
        self.learning_rate = 0.001

        # Define encoder layer
        self.encoder_attention = tf.keras.layers.Attention()
        self.encoder_norm = tf.keras.layers.LayerNormalization(axis=-1)

        # Define decoder layer
        self.decoder = tf.keras.layers.Attention()

    @tf.function
    def call(self, originals):
        """
        :param originals: sliced spectrograms of the original two songs that will make up a mashup
        :return artif_mash: The spectrogram of the mashup generated from the model
        """

        songs_encoded = self.encoder(originals)
        artif_mash = self.decoder(songs_encoded)

        return artif_mash

    def cross_correlation(self, artif, real):
        """
        Calculates cross correltaion between two spectrograms.

        :param artif: artificial spectrogram generated from model
        :param real: real spectrogram downloaded from YouTube
        :return cross_corr: cross correlation between the two spectrograms
        """
        return 1

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        raw_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
        masked_loss = raw_loss * mask
        return tf.reduce_sum(masked_loss)


    def __call__(self, *args, **kwargs):
        return super(SongMasher, self).__call__(*args, **kwargs)
