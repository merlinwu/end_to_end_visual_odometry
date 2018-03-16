import tensorflow as tf
import model
import numpy as np

# tf.logging.set_verbosity(1)
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# sess = tf.Session(config=config)

max_timesteps = 10
batch_size = 8

input_width = 1280
input_height = 384
input_channels = 6

inputs = tf.placeholder(tf.float32, name="inputs",
                        shape=[batch_size, max_timesteps, input_width, input_height, input_channels])
se3_labels = tf.placeholder(tf.float32, name="se3_labels",
                            shape=[batch_size, max_timesteps, input_width, input_height, input_channels])
fc_labels = tf.placeholder(tf.float32, name="se3_labels",
                           shape=[batch_size, max_timesteps, input_width, input_height, input_channels])

fc_outputs, se3_outputs = model.build_model(inputs)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# =========== Visualization ============
writer = tf.summary.FileWriter('graph_viz/')
writer.add_graph(tf.get_default_graph())
sess_ret = sess.run(se3_outputs, {inputs: np.random.random([8, 10, 1280, 384, 6])})

sess.close()
