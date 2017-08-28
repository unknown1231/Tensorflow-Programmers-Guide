import tensorflow as tf 

# Feedable Iterator

training_dataset = tf.contrib.data.Dataset.range(100).map(lambda x : x + tf.random_uniform(shape = [], minval = -10, maxval = 10, dtype = tf.int64)).repeat()
validation_dataset = tf.contrib.data.Dataset.range(50)

handle = tf.placeholder(tf.string, shape = [])
iterator = tf.contrib.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

with tf.Session() as sess:
	training_handle = sess.run(training_iterator.string_handle())
	validation_handle = sess.run(validation_iterator.string_handle())
    
    # the loop will be infinite
	while True:
		for _ in range(200):
			sess.run(next_element, feed_dict = {handle : training_handle})

		sess.run(validation_iterator.initializer)
		for _ in range(50):
			sess.run(next_element, feed_dict = {handle : validation_handle})

		

