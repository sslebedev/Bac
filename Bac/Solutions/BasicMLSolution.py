from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

from BatchGenerator import generator
from InitData import init_data

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

lookback = 300
batch_size = 64

tds, vd = init_data(100)
val_data = vd
val_gen = generator(val_data,
                    lookback=lookback,
                    shuffle=False,
                    batch_size=batch_size)
val_steps = (1000 - lookback) // batch_size

# ---------------------------------------------

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback, tds[0].shape[-1])))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(), loss='binary_crossentropy')

loss = None
val_loss = None
epochs = None

tgs = [generator(train_data,
                 lookback=lookback,
                 shuffle=True,
                 batch_size=batch_size) for train_data in tds]


for tg in tgs:
    history = model.fit_generator(tg,
                                  steps_per_epoch=(1000 - 300) // batch_size,
                                  epochs=5,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    if loss is None:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
    else:
        loss += history.history['loss']
        val_loss += history.history['val_loss']
        epochs = [*epochs, *range(epochs[-1] + 1, epochs[-1] + 1 + len(history.history['loss']))]

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show(block=True)
