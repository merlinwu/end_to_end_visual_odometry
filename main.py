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

input_data, poses_quat, poses_ypr_w_covar = model.build_model(batch_size, max_timesteps)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# =========== Visualization ============
writer = tf.summary.FileWriter('graph_viz/')
writer.add_graph(tf.get_default_graph())
sess_ret = sess.run(poses_quat, {input_data: np.random.random([8, 10, 1280, 384, 6])})

sess.close()
