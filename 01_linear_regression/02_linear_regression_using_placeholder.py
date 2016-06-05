import tensorflow as tf


'''
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

import tensorflow as tf
rand_var = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
oper = sess.run(rand_var)
print oper

[[-0.10988665  0.05830932]]
'''
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# y_data = W * x_data + b
hypothesis = W * X + b


'''
import tensorflow as tf
x = [[1.,4.],[2., 3.]]

rm0 = tf.reduce_mean(x) 
rm1 = tf.reduce_mean(x, 0) 
rm2 = tf.reduce_mean(x, 1) 
sess = tf.Session()
print sess.run(rm0)
print sess.run(rm1)
print sess.run(rm2)

2.5
[ 1.5  3.5]
[ 2.5  2.5]
'''
cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1) # Learning rate
optimizer = tf.train.GradientDescentOptimizer(a);
train = optimizer.minimize(cost)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# training data
x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

# Fit the line.
for step in xrange(601):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
 #   if step % 20 == 0:
 #       print(step, sess.run(W), sess.run(b))

print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.5})