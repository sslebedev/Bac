import numpy as np
from matplotlib import pyplot as plt

from SampleGenerator import init_test_samples
from Utility import deserialize_model

samples = init_test_samples()
model = deserialize_model()

plt.figure()
plt.plot([[sample[-1][0]] for sample in samples], label='Channel 0')
plt.plot([[sample[-1][1]] for sample in samples], label='Channel 1')
predictions = [[model.predict(np.array(sample)) for sample in samples]]
plt.plot(predictions, label='Probability')

plt.plot([1] * len(samples))
plt.plot([0.5] * len(samples))

plt.title('Prediction on samples')
plt.legend()
plt.show(block=True)