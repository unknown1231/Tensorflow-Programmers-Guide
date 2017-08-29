import tensorflow as tf 

# decoding into a dense tensor and resizing to a fixed shape
def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	# decoding
	image_decoded = tf.image.decode_image(image_string)# 0-D string
	# resizing
	image_resize = tf.image.resize_image_with_crop_or_pad(image_decoded, 28, 28)# image, target_width, target_heigth
	print(image_resize)
	return image_resize, label # if not there, "none values supported" error occurs.

filenames = tf.constant(['IMG_3832.JPG', 'IMG_4953.JPG'])
labels = tf.constant([0, 37])
dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels)).map(_parse_function)