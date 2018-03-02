import os
import numpy as np
import tensorflow as tf

# tf.logging.set_verbosity(1)
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# sess = tf.Session(config=config)

max_timesteps = 10
batch_size = 8
input_width = 40
input_height = 12
input_channels = 512
pose_size = 13
input_data = tf.placeholder(tf.float32, name="input_data",
                            shape=[batch_size, max_timesteps, input_width, input_height, input_channels])
input_data_length = tf.placeholder(tf.float32, name="input_data_length", shape=[batch_size])

output_data = tf.placeholder(tf.float32, name="outputs", shape=[max_timesteps, batch_size, pose_size])


# ================= Build Graph Model ===================
# CNN Block

def cnn_model(inputs):
    # conv6 = tf.layers.conv2d(inputs, name="Conv6", filters=1024, kernel_size=(3, 3,), strides=(2, 2), padding="same")
    conv6 = tf.contrib.layers.conv2d(inputs, num_outputs=1024, kernel_size=(3, 3,),
                                     stride=(2, 2), padding="same", scope="conv_6")
    return conv6


def fc_model(inputs):
    fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu)
    fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=tf.nn.relu)
    return fc_12


# pose 1 in xyz + rpy
# pose 2 in xyz + quaternion
# returns quaternion
def pose_comp(pose_1, pose_2):
    return tf.concat([pose_1, [0]], 0) + pose_2


def calc_covar(input):
    return input

def se3_and_covar(rpy_pose_with_covar, quat_pose_with_covar):
    pass


with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):
    input_data = tf.unstack(input_data, axis=1)
    cnn_outputs = tf.map_fn(cnn_model, input_data, dtype=tf.float32, name="cnn_map")

cnn_outputs = tf.reshape(cnn_outputs,
                         [batch_size, max_timesteps,
                          cnn_outputs.shape[2] * cnn_outputs.shape[3] * cnn_outputs.shape[4]])

# RNN Block
with tf.variable_scope("RNN"):
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
    layered_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    rnn_outputs, _ = tf.nn.dynamic_rnn(layered_lstm_cell, cnn_outputs, dtype=tf.float32)

with tf.variable_scope("Fully-Connected", reuse=tf.AUTO_REUSE):
    rnn_outputs = tf.unstack(rnn_outputs, axis=1)
    fc_outputs = tf.map_fn(fc_model, rnn_outputs, dtype=tf.float32, name="fc_map")

with tf.variable_scope("SE3"):
    rnn_outputs = tf.unstack(rnn_outputs, axis=0) # unstack the batches for processing along the timesteps

tf.foldr

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# =========== Visualization ============
writer = tf.summary.FileWriter('graph_viz/')
writer.add_graph(tf.get_default_graph())
# sess_ret = sess.run(x, {inputs: np.random.random([1, 40, 12, 512])})

sess.close()
