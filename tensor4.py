import tensorflow as tf 

# weights = tf.random_normal(...)

with tf.device('/device:CPU:0'):
	image1 = tf.image.decode_jpeg(tf.read_file('img_3832.jpg'))
	print(image1)

#with tf.device('/device:GPU:0'):
	#multi = tf.matmul(weights, image1)
	#print(multi)