import os
import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, name="inputs", shape=[None, 40, 12, 512])
cnn_model = tf.layers.conv2d(x, name="Conv6", filters=1024, kernel_size=(3, 3,), strides=(2, 2), padding="same")


# =========== Visualize the model for tensor board ============
writer = tf.summary.FileWriter('graph_viz/')
writer.add_graph(tf.get_default_graph())
# ==============================================================

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess_ret = sess.run(cnn_model, {x: np.random.random([1, 40, 12, 512])})

print(sess_ret.shape)

sess.close()
