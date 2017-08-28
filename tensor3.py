import tensorflow as tf 

# tf.Graph and Session

c_0 = tf.constant(0, name = 'c') # operation name is c
print(c_0)
c_1 = tf.constant(2, name = 'c') # operation name is c_1
print(c_1)

# using CPU with tf.device 
with tf.device('/device:CPU:0'):
	# outer scope
	with tf.name_scope('outer'):
		c_2 = tf.constant(2, name = 'c') # opeartion name is outer/c
		print(c_2)

		with tf.name_scope('inner'):
			# inner scope
			c_3 = tf.constant(3, name = 'c') # operation name is outer/inner/c
			print(c_3)
		c_4 = tf.constant(4, name = 'c') # operation name is outer/c_1
		print(c_4)

		with tf.name_scope('inner'):
			c_5 = tf.constant(5, name = 'c') # operation name is outer/inner/c_1
			print(c_5)
	

	    

	




    
	


	

	








	

    
    

	








