import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)


with tf.Session() as sess:
	print "a=2, b=3"
	print "Addition with constants: %d" % sess.run(a+b)
	print "Multiplication with constants: %d" % sess.run(a*b)

# with statement
# 
# with open('output.txt', 'w') as f:
# 	 f.write('Hi there!')
# 
# The advantage of using a with statement
# is that it is guaranteed to close the file 
# no matter how the nested block exits.