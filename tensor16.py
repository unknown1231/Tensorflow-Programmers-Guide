import tensorflow as tf 
import numpy as np
import pandas as pd 

#np.save('/tmp/test_np', np.array([[1, 2, 3], [1, 2, 3]]))
data = pd.read_csv('test_np.csv') 
features = data['features']
labels = data['lables']
print(features[1])# just to test

assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
print(dataset)
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
	print(sess.run(iterator.initializer, feed_dict = {features_placeholder : features, labels_placeholder : labels}))


