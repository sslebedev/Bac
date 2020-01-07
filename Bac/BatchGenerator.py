# Test data generation
import numpy as np


def generator(data, history_len, delay=0, min_index=0, max_index=None,
              shuffle=False, batch_size=128, step=6):
    # data: The original array of normalized floating point data.
    # lookback: How many timesteps back should our input data go.
    # delay: How many timesteps in the future should our target be.
    # min_index and max_index: Indices in the data array that delimit which timesteps to draw from.
    #       This is useful for keeping a segment of the data for validation and another one for testing.
    # shuffle: Whether to shuffle our samples or draw them in chronological order.
    # batch_size: The number of samples per batch.
    # step: The period, in timesteps, at which we sample data.
    #       We will set it 6 in order to draw one data point every hour.
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + history_len

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + history_len, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + history_len
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            history_len // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - history_len, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
