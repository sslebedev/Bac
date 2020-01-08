import numpy as np

from DataGenerators import DataGeneratorBase


def flow_to_data(flow: DataGeneratorBase):
    lines = flow[0:len(flow)]
    float_data = np.zeros((len(lines), flow.dim()))
    for i, line in enumerate(lines):
        float_data[i, :] = line
    return float_data


def generator(data, lookback,
              shuffle=False, batch_size=128):
    # get_next_batch: next batch data generator
    # history_len: How many timesteps back should our input data go.
    # total_samples: How many samples to make
    # shuffle: Whether to shuffle our samples or draw them in chronological order.
    # batch_size: The number of samples per batch.
    delay = 0
    step = 1
    max_index = len(data) - delay - 1
    min_index = 0
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
