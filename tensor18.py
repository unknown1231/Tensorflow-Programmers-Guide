import tensorflow as tf 

# two text files
filenames = ["text1.txt", "text2.txt"]
file_len = len(filenames)
# TextLineDataset is an easy way of extracting text from one or more text files
dataset = tf.contrib.data.TextLineDataset(filenames)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer)
	for _ in data_len:
		print(sess.run(next_element))

	
''' Flaws in the Above code :-
since filename length is 2 it iterates twice. if text1 contains 2 lines, text2 will not get printed.
'''

# another program
# in this program the line with '#' is skipped and the first line i skipped i think.
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(filenames)
file_len = len(filenames)
dataset1 = dataset1.flat_map(lambda files : tf.contrib.data.TextLineDataset(filenames).skip(1).filter(
	lambda line : tf.not_equal(tf.substr(line, 0, 1), '#')))