import tensorflow as tf 

# example proto is a scalar string. it is paired into scalar string(image) and scalar integers(label)
def _parse_function(example_proto):
	# FixedLenFeature treats a sparse input as a dense input
	features = { image : tf.FixedLenFeature((), tf.string, default_value = "")
	             label: tf.FixedLenFeature((, tf.int32, default_value = 0))}
	parses_feature = tf.parses_single_example(example_proto, features)
	return parses_feature['image'], parses_feature['label']

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecord(filenames).map(_parse_function)
