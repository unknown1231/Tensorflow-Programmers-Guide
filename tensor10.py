import tensorflow as tf 

# from_tensor_slices(tensors) creates a dataset whose elements are slices of the given tensors
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types, '---->DataSet1')# returns type of each component of an element in the dataset
print(dataset1.output_shapes, '---->DataSet1')# returns shape of each component of an element in the dataset

dataset2 = tf.contrib.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100], maxval = 100, dtype = 'int32')))
print(dataset2.output_types, '---->DataSet2')
print(dataset2.output_shapes, '---->DataSet2')

dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types, '---->DataSet3')
print(dataset3.output_shapes, '---->DataSet3')

# from_tensors(tensors) creates a dataset with a single element comprising  the given tensors
dataset4 = tf.contrib.data.Dataset.from_tensors((tf.random_uniform([4, 10]), tf.random_uniform([4, 100], maxval = 100, dtype = 'int32')))
print(dataset4.output_types, '---->DataSet4')
print(dataset4.output_shapes, '---->DataSet4')

# OR ----------------------------------------------------------------------

dataset = tf.contrib.data.Dataset.from_tensor_slices(
	 {"a" : tf.random_uniform([4]),
	  "b" : tf.random_uniform([4, 100], maxval = 100, dtype = 'int32')})
print(dataset.output_types, '---->Dataset')
print(dataset.output_shapes, '---->Dataset')
