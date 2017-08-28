import tensorflow as tf 

# One Shot Iterator
dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	for i in range(100):
		value = sess.run(next_element)
		print(value)
		assert i == value # asserts i and value i.e it checks wether both satisfy the given condition which is '=='
	    

	