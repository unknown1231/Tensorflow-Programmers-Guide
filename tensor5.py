import tensorflow as tf 

with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.get_variable('w_0', shape = (2, 2, 3), initializer = tf.zeros_initializer())
  b_0 = tf.get_variable('b_0', shape = (2, 2, 3), initializer = tf.zeros_initializer())
  w_1 = tf.get_variable('w_1', shape = (2, 3, 3), initializer = tf.zeros_initializer())
  b_1 = tf.get_variable('b_1', shape = (2, 2, 3), initializer = tf.zeros_initializer())

  input_data = tf.placeholder('float', shape = [2, 1])     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"