import tensorflow as tf 

get_var1 = tf.get_variable('get_var1', shape = ())
get_var2 = get_var1.assign(10)
output_var = get_var2 + 5

with tf.Session() as sess:
	sess.run(get_var1.initializer)
	result, _= sess.run(output_var, get_var2)
	print(result)
