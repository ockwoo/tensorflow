import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session as sess:
	print "Addition with variable: %i" % sess.run(add, feed_dict={a: 2, b: 3})
	print "Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3})

