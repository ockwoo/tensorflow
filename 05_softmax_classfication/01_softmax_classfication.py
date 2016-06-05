import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_5.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print x_data
print y_data
'''
import numpy as np

xy = np.loadtxt('train_5.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print x_data
print y_data

[[ 1.  2.  1.]
 [ 1.  3.  2.]
 [ 1.  3.  4.]
 [ 1.  5.  5.]
 [ 1.  7.  5.]
 [ 1.  2.  5.]
 [ 1.  6.  6.]
 [ 1.  7.  7.]]
[[ 0.  0.  1.]
 [ 0.  0.  1.]
 [ 0.  0.  1.]
 [ 0.  1.  0.]
 [ 0.  1.  0.]
 [ 0.  1.  0.]
 [ 1.  0.  0.]
 [ 1.  0.  0.]]


import numpy as np

xy = np.loadtxt('train_5.txt', unpack=True, dtype='float32')
x_data = xy[0:3]
y_data = xy[3:]

print x_data
print y_data

[[ 1.  1.  1.  1.  1.  1.  1.  1.]
 [ 2.  3.  3.  5.  7.  2.  6.  7.]
 [ 1.  2.  4.  5.  5.  5.  6.  7.]]
[[ 0.  0.  0.  0.  0.  0.  1.  1.]
 [ 0.  0.  0.  1.  1.  1.  0.  0.]
 [ 1.  1.  1.  0.  0.  0.  0.  0.]]
'''
X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.zeros([3, 3]))


hypothesis = tf.nn.softmax(tf.matmul(X, W))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis), reduction_indices=1) 
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


learning_rate = tf.Variable(0.001) # Learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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

a = sess.run(hypothesis, feed_dict={X:[[1,11,7]]})
print a, sess.run(tf.arg_max(a, 1))

b = sess.run(hypothesis, feed_dict={X:[[1,3,4]]})
print a, sess.run(tf.arg_max(b, 1))