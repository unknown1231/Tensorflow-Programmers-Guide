import tensorflow as tf

get_var1 = tf.get_variable('get_var1', shape = (), initializer = tf.zeros_initializer())
assignment = get_var1.assign(1)
print(assignment)
#get_var2 = tf.get_variable('get_var2', initializer = get_var1.initialized_value() + 1)
with tf.control_dependencies([assignment]):
	get_var2 = get_var1.read_value()
	get_var3 = get_var1.read_value()
	get_var4 = get_var1.read_value()
	print(get_var2)
	print( get_var3)
	print( get_var4)

#tf.add_to_collection('get_collection_var', [get_var1, get_var2])
#get_collection_var = tf.get_collection('get_collection_var')
#print(get_collection_var)
#print('\n')
#print(get_var1, '\n')
#print(get_var2)

#with tf.Session() as sess:
#	print('before init \n')
#	print(sess.run(tf.report_uninitialized_variables()))
#	print('\n after init \n')
#	print(sess.run(tf.global_variables_initializer()))

