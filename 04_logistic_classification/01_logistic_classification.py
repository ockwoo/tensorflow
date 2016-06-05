import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_4.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]
'''
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]  # == xy[0:3]
y_data = xy[-1]     # == xy[3]

print x_data
print y_data

[[ 1.  1.  1.  1.  1.]
 [ 1.  0.  3.  0.  5.]
 [ 0.  2.  0.  4.  0.]]
[ 1.  2.  3.  4.  5.]

'''

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))

h = tf.matmul(W,X)
#################################################################
hypothesis = tf.div(1. , 1.+tf.exp(-h)) # Sigmoid function H(X) = 1 / 1 + e^(-h) 
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)) 
################################################################

a = tf.Variable(0.1) # Learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 50 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print 'Test'
print '-----------------------------------'

print sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5
print sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5

print sess.run(hypothesis, feed_dict={X: [[1,1], [4,3], [3,5]]}) > 0.5
