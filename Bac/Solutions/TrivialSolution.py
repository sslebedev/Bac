import numpy as np
from matplotlib import pyplot as plt

from BatchGenerator import generator
from InitData import init_data

lookback = 300
batch_size = 128

tds, vd = init_data(1)
val_data = vd
val_gen = generator(val_data,
                    lookback=lookback,
                    shuffle=False,
                    batch_size=batch_size)
val_steps = (1000 - lookback) // batch_size

# plot
x1 = val_data[:, 1]
x2 = val_data[:, 2]
plt.figure()
plt.plot(range(len(x1)), x1)
plt.plot(range(len(x2)), x2)
plt.legend(['temp', 'pres'])
plt.show(block=True)


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)


print("Naive: %f" % evaluate_naive_method())