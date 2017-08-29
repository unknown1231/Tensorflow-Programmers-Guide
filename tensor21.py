import tensorflow as tf 
# Batching dataset elements
# Simple Batching

inc_dataset = tf.contrib.data.Dataset.range(100)
dec_dataset = tf.contrib.data.Dataset.range(0, -100, -1)
dataset = tf.contrib.data.Dataset.zip((inc_dataset, dec_dataset))
batch_dataset = dataset.batch(4)

iterator = batch_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	for _ in range(3):
		print(sess.run(next_element))