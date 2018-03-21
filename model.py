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
    initial_pose = [0, 0, 0,
                    1, 0, 0, 0]  # position + orientation in quat
    poses = tools.foldl(se3.se3_comp, fc_timesteps[:, 0:7], name="se3_comp_foldl",
                        initializer=tf.constant(initial_pose, dtype=tf.float32), dtype=tf.float32)
    return poses


def build_model(inputs, lstm_init_state):
    with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):
        inputs_unstacked = tf.unstack(inputs, axis=1)
        cnn_outputs = tf.map_fn(cnn_model, inputs_unstacked, dtype=tf.float32, name="cnn_map")

    cnn_outputs = tf.reshape(cnn_outputs, [cnn_outputs.shape[0], cnn_outputs.shape[1],
                                           cnn_outputs.shape[2] * cnn_outputs.shape[3] * cnn_outputs.shape[4]])

    # RNN Block
    with tf.variable_scope("RNN"):
        # setup the initial state for LSTM
        state_per_layer_list = tf.unstack(lstm_init_state, axis=0)
        lstm_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(2)]
        )

        # setup the RNN layers
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=1024)
        layered_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(layered_lstm_cell, cnn_outputs, dtype=tf.float32)

    with tf.variable_scope("Fully-Connected", reuse=tf.AUTO_REUSE):
        lstm_outputs = tf.unstack(lstm_outputs, axis=1)
        fc_outputs = tf.map_fn(fc_model, lstm_outputs, dtype=tf.float32, name="fc_map")

    with tf.variable_scope("SE3"):
        # at this point the outputs from the fully connected layer are  [x, y, z, yaw, pitch, roll, 6 x covars]
        se3_outputs = tf.map_fn(se3_comp_over_timesteps, fc_outputs, dtype=tf.float32, name="se3_map")

    return fc_outputs, se3_outputs, lstm_states
