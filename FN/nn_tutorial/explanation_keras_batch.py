import numpy as np
from tensorflow.python import keras as K

# 2-layer neural network.
model = K.Sequential([
    K.layers.Dense(units=4, input_shape=((2, )),
                   activation="sigmoid"),
    K.layers.Dense(units=4),
])

# Make batch size = 3 data (dimension of x is 2).
batch = np.random.rand(3, 2)

y = model.predict(batch)
print(y.shape)  # Will be (3, 4)
