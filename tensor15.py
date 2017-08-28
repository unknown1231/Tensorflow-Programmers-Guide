import tensorflow as tf 

dataset = tf.contrib.data.Dataset.range(10)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

result = tf.add(next_element, next_element)

with tf.Session() as sess:
	sess.run(iterator.initializer)
	for _ in range(10):
		print(sess.run(result))


	try:
		sess.run(result)
	except tf.errors.OutOfRangeError:
		print('end of DataSet!!')
	