import tensorflow as tf 

# processing multiple epochs and random shuffling input data
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames).map(lambda x : x + filenames).shuffle(buffer_size = 10000).batch(32).repeat()


iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	for _ in range(100):
		sess.run(iterator.initializer)
		while True:
			try:
				sess.run(next_element)
			except tf.errors.OutOfRangeError:
				break

'''
This program wont work
'''