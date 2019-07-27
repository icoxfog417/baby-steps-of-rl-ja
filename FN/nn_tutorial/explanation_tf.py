import numpy as np
import tensorflow as tf

# Weight (row=4 x col=2).
a = tf.Variable(np.random.rand(4, 2))

# Bias (row=4 x col=1).
b = tf.Variable(np.random.rand(4, 1))

# Input(x) (row=2 x col=1).
x = tf.compat.v1.placeholder(tf.float64, shape=(2, 1))

# Output(y) (row=4 x col=1).
y = tf.matmul(a, x) + b


with tf.Session() as sess:
    # Initialize variable.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Make input to x.
    x_value = np.random.rand(2, 1)

    # Execute culculation.
    y_output = sess.run(y, feed_dict={x: x_value})
    print(y_output.shape)  # Will be (4, 1)
