import tensorflow as tf 

place_holder = tf.placeholder(dtype = 'int64', shape = [])
#place_holder1 = tf.placeholder(place_holder, dtype = 'float64', shape = [])
dataset = tf.contrib.data.Dataset.range(place_holder)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer, feed_dict = {place_holder : 10})
	for i in range(10):
		value = sess.run(next_element)
		print(value)
		assert i == value
print('-------------')

with tf.Session() as sess:
	sess.run(iterator.initializer, feed_dict ={place_holder : 100})
	for i in range(100):
		value = sess.run(next_element)
		print(value)
		assert i == value
	
	

