import tensorflow as tf


tf.enable_eager_execution()
tfe = tf.contrib.eager


def f(x, a, b):
    return tf.add(tf.multiply(x, a), b)


def grad(f):
    return lambda x, a, b: tfe.gradients_function(f)(x, a, b)


x = 2.0
a = 3.0
b = -1.0

print(f(x, a, b))
print(grad(f)(x, a, b))
