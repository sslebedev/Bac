# Original from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, samples, batch_size=32, lookback=50, n_channels=2, shuffle=True):
        # Initialization
        self.lookback = lookback
        self.batch_size = batch_size
        self.samples = samples
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        samples_temp = [self.samples[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(samples_temp)

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, samples_temp):
        # Generates data containing batch_size samples
        # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, self.lookback, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, sample in enumerate(samples_temp):
            # As sample is a pair (data, label)
            # Store sample
            x[i, ] = sample[0]

            # Store class
            y[i] = sample[1]

        return x, y