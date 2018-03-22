import tensorflow as tf
import se3
import tools


# CNN Block
def cnn_model(inputs):
    conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(7, 7,),
                                      stride=(2, 2), padding="same", scope="conv_1")
    conv_2 = tf.contrib.layers.conv2d(conv_1, num_outputs=128, kernel_size=(5, 5,),
                                      stride=(2, 2), padding="same", scope="conv_2")

    conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=256, kernel_size=(5, 5,),
                                      stride=(2, 2), padding="same", scope="conv_3")
    conv_3_1 = tf.contrib.layers.conv2d(conv_3, num_outputs=256, kernel_size=(3, 3,),
                                        stride=(1, 1), padding="same", scope="conv_3_1")

    conv_4 = tf.contrib.layers.conv2d(conv_3_1, num_outputs=512, kernel_size=(3, 3,),
                                      stride=(2, 2), padding="same", scope="conv_4")
    conv_4_1 = tf.contrib.layers.conv2d(conv_4, num_outputs=512, kernel_size=(3, 3,),
                                        stride=(1, 1), padding="same", scope="conv_4_1")

    conv_5 = tf.contrib.layers.conv2d(conv_4_1, num_outputs=512, kernel_size=(3, 3,),
                                      stride=(2, 2), padding="same", scope="conv_5")
    conv_5_1 = tf.contrib.layers.conv2d(conv_5, num_outputs=512, kernel_size=(3, 3,),
                                        stride=(1, 1), padding="same", scope="conv_5_1")

    conv_6 = tf.contrib.layers.conv2d(conv_5_1, num_outputs=1024, kernel_size=(3, 3,),
                                      stride=(2, 2), padding="same", scope="conv_6")
    return conv_6


def fc_model(inputs):
    fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu)
    fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=tf.nn.relu)
    return fc_12


def se3_comp_over_timesteps(fc_timesteps):
    # position + orientation in quat
    initial_pose = tf.constant([0, 0, 0, 1, 0, 0, 0], tf.float32)

    poses = []
    pose = initial_pose
    fc_timesteps = tf.unstack(fc_timesteps[:, 0:7], axis=1)  # take the x, y, z, y, p, r
    for d_pose in fc_timesteps:
        pose = se3.se3_comp(pose, fc_t)
        poses.append(pose)
    return tf.stack(poses)


def cudnn_lstm_unrolled(inputs, initial_state):
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(2, 1024)
    outputs, final_state = lstm(inputs, initial_state=initial_state)
    return outputs, final_state


def build_model(inputs, lstm_init_state):
    with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):
        cnn_outputs = tools.static_map_fn(cnn_model, inputs, axis=0)

    cnn_outputs = tf.reshape(cnn_outputs, [cnn_outputs.shape[0], cnn_outputs.shape[1],
                                           cnn_outputs.shape[2] * cnn_outputs.shape[3] * cnn_outputs.shape[4]])

    # RNN Block
    with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
        lstm_outputs, lstm_states = cudnn_lstm_unrolled(cnn_outputs, lstm_init_state)

    with tf.variable_scope("Fully-Connected", reuse=tf.AUTO_REUSE):
        fc_outputs = tools.static_map_fn(fc_model, lstm_outputs, axis=0)

    with tf.variable_scope("SE3", reuse=tf.AUTO_REUSE):
        # at this point the outputs from the fully connected layer are  [x, y, z, yaw, pitch, roll, 6 x covars]
        se3_outputs = tools.static_map_fn(se3_comp_over_timesteps, fc_outputs, axis=1)

    return fc_outputs, se3_outputs, lstm_states
