import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X


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


#a = tf.Variable(0.1) # Learning rate
#optimizer = tf.train.GradientDescentOptimizer(a);
#train = optimizer.minimize(cost)

descent = W - tf.mul(0.1 , tf.reduce_mean( tf.mul((tf.mul(W,X) - Y), X)))
update = W.assign(descent)
'''
import tensorflow as tf

x = tf.Variable(0)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

oper = x.assign(1)
print(sess.run(oper))

1
'''

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# training data
x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

# Fit the line.
for step in xrange(50):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
