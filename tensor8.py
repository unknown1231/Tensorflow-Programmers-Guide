import tensorflow as tf 

x = tf.placeholder(tf.float32, shape = [3])
y = tf.square(x)

with tf.Session() as sess:
	# feeding the placeholder a value and giving it to y
	xx = sess.run(y, feed_dict = {x : [1.0, 2.0, 3.0]})
	print(xx)

	# sess.run(y) gives an error because x is a placeholder and no value is assigned to it
	# sess.run(y, feed_dict = {x: 7.0}) raises an error because the shape of x does not match the placeholder