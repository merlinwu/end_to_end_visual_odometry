import os
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(1)
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# sess = tf.Session(config=config)

max_sequence_count = 1
batch_size = 1

# ================= Build Graph Model ===================
# CNN Block
# inputs = tf.placeholder(tf.float32, name="inputs", shape=[batch_size, 40, 12, 512])
# x = tf.layers.conv2d(inputs, name="Conv6", filters=1024, kernel_size=(3, 3,), strides=(2, 2), padding="same")

# flatten CNN to feed into RNN
# x = tf.reshape(x, shape=[batch_size, max_sequence_count, int(x.shape[1] * x.shape[2] * x.shape[3]) // max_sequence_count])
inputs = tf.placeholder(tf.float32, name="inputs", shape=[1, 1, int(122880)])

lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
# lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
layered_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
x = tf.nn.dynamic_rnn(layered_lstm_cell, inputs, dtype=tf.float16)

# RNN Block

# =========== Visualization ============
writer = tf.summary.FileWriter('graph_viz/')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# sess_ret = sess.run(x, {inputs: np.random.random([1, 40, 12, 512])})

sess.close()
