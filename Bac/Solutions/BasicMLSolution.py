from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

from DataGenerator import DataGenerator

from SampleGenerator import init_data

# --------------------------------------------------
from Utility import serialize_model, deserialize_model

LOOKBACK = 100
CHANNELS = 2

model = Sequential()
model.add(layers.Flatten(input_shape=(LOOKBACK, CHANNELS)))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(), loss='binary_crossentropy')

# --------------------------------------------------

ts, vs = init_data(200, 40)

training_generator = DataGenerator(ts, 32, lookback=LOOKBACK, shuffle=True)
validation_generator = DataGenerator(vs, 32, lookback=LOOKBACK, shuffle=False)

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              epochs=20)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

# --------------------------------------------------

serialize_model(model)

# --------------------------------------------------

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show(block=True)

