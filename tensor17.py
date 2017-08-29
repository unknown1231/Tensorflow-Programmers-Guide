import tensorflow as tf 

filenames = tf.placeholder(tf.string, shape = [None]) #None means that it has no value and it can take any value....leaving it empty means its 0
# Parsing data into tensors
dataset = tf.contrib.data.TFRecordDataset(filenames).map(lambda x : x + filenames).repeat().batch(32)
# iterator
iterator = dataset.make_initializable_iterator()

# TFRecord stores the files in binary format 
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
validation_filenames = ["test_np.tfrecord"]

with tf.Session() as sess:
	print(sess.run(iterator.initializer, feed_dict = {filenames : training_filenames}))
	print(sess.run(iterator.initializer, feed_dict = {filenames : validation_filenames}))
