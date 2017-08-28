# dont run this program!!
import tensorflow as tf 

y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
	options = tf.RunOptions()
	options.output_partition_graphs = True
	options.trace_level = tf.RunOptions.FULL_TRACE

	metadata = tf.RunMetadata()
	sess.run(y, options = options, run_metadata = metadata)

	print(metadata.partion_graphs)

	print(metadata.step_stats)

	# dont run this program!!