import numpy as np
from tensorflow.python import keras as K

model = K.Sequential([
    K.layers.Dense(units=4, input_shape=((2, ))),
])

weight, bias = model.layers[0].get_weights()
print("Weight shape is {}.".format(weight.shape))
print("Bias shape is {}.".format(bias.shape))

x = np.random.rand(1, 2)
y = model.predict(x)
print("x is ({}) and y is ({}).".format(x.shape, y.shape))
