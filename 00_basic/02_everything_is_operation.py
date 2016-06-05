import tensorflow as tf

# Start Session
sess = tf.Session()

# Basic constant operation
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

c = a + b

print sess.run(c)