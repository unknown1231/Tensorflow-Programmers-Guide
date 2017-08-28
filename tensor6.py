import tensorflow as tf 


a = tf.Variable(5)
b = tf.Variable(5)
c = tf.Variable(5)
d = tf.Variable(5)
place_holder = tf.placeholder(dtype = 'int32')     # placed on "/job:worker"
result = a * b * c * d * place_holder

init = tf.global_variables_initializer()

with tf.Session() as sess :
	sess.run(init)
	multi = sess.run(result, feed_dict = {place_holder : 5})
	print(multi)

