import numpy as np
from matplotlib import pyplot as plt

from SampleGenerator import init_test_samples
from Utility import deserialize_model

samples, labels = init_test_samples()
model = deserialize_model()

t = range(100, 1000)

plt.figure()
plt.plot(t, [[sample[-1][0]] for sample in samples], label='Channel 0')
plt.plot(t, [[sample[-1][1]] for sample in samples], label='Channel 1')
plt.plot(t, labels, label='Hasard initiated')
predictions = model.predict(np.array(samples))
plt.plot(t, predictions, label='Probability')

plt.plot(t, [1] * len(samples), 'g--')
plt.plot(t, [0.5] * len(samples), 'g--')

plt.title('Prediction on samples')
plt.legend()
plt.show(block=True)