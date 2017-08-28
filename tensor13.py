import tensorflow as tf 

epochs = 20

# ReInitializable Iterator
# output_types and output_shapes are the same type and shape
# lamda is nothing....it becomes random uniform when its added with x...the minval lamda(x) can have is -10 and the maxval is 10
# these lambda values are added to the range(100)
# the lambda values are randomly added during every epoch
training_dataset = tf.contrib.data.Dataset.range(100).map(lambda x : x + tf.random_uniform(shape = [], minval = -10, maxval = 10, dtype = tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)

iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
print(training_dataset.output_types)
print(training_dataset.output_shapes)
print(validation_dataset.output_types)
print(validation_dataset.output_shapes)
next_element = iterator.get_next()

# reinitializing
# iterator is used for both training and validation. it is possible since shapes and types are equal for both
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
	for _ in range(epochs): # the entire loop runs 20 times
		sess.run(training_init_op)
		for i in range(100):
			sess.run(next_element)

		sess.run(validation_init_op)
		for i in range(50):
			sess.run(next_element)




