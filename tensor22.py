import tensorflow as tf 

# batching tensors with padding

dataset = tf.contrib.data.Dataset.range(100)
'''tf.fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
'''
dataset = dataset.map(lambda x : tf.fill([tf.cast(x, tf.int32)], x)) # cast can change the type of the tensor
dataset = dataset.padded_batch(4, padded_shapes = [-1]) # or padded_shapes = [None]

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	for _ in range(2):
		print(sess.run(next_element))