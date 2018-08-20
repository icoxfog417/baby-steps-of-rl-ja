import numpy as np
from tensorflow.python import keras as K

model = K.Sequential([
    K.layers.Dense(units=4, input_shape=((2, ))),
])

# Make batch size = 3 data (dimension of x is 2).
batch = np.random.rand(3, 2)

y_output = model.predict(batch)
print(y_output.shape)  # Will be (3, 4)
