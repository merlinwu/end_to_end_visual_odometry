import tensorflow as tf
import model
import losses
import numpy as np

# tf.logging.set_verbosity(1)
# config = tf.ConfigProto()
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# sess = tf.Session(config=config)

# =================== CONFIGS ========================
timesteps = 8
batch_size = 4

input_width = 1280
input_height = 384
input_channels = 6

num_epochs = 50

k = 10

# =================== INPUTS ========================
# All time major
inputs = tf.placeholder(tf.float32, name="inputs",
                        shape=[timesteps, batch_size, input_width, input_height, input_channels])

# init LSTM states, 2 layers, 2 (cell + hidden states), batch size, and 1024 state size
lstm_init_state = tf.placeholder(tf.float32, name="lstm_init_state", shape=[2, 2, batch_size, 256])

# 7 for translation + quat
se3_labels = tf.placeholder(tf.float32, name="se3_labels", shape=[timesteps, batch_size, 7])

# 6 for translation + rpy, labels not needed for covars
fc_labels = tf.placeholder(tf.float32, name="se3_labels", shape=[timesteps, batch_size, 6])

# dynamic learning rates
se3_lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
fc_lr = tf.placeholder(tf.float32, name="fc_lr", shape=[])

# =================== MODEL + LOSSES + Optimizer ========================
fc_outputs, se3_outputs, lstm_states = model.build_model(inputs, lstm_init_state)

with tf.variable_scope("LOSSES"):
    se3_losses = losses.se3_losses(se3_outputs, se3_labels, k)
    fc_losses = losses.fc_losses(fc_outputs, fc_labels)

with tf.variable_scope("TRAINER"):
    se3_trainer = tf.train.AdamOptimizer(learning_rate=se3_lr).minimize(se3_outputs)
    fc_trainer = tf.train.AdamOptimizer(learning_rate=fc_lr).minimize(fc_outputs)

# =================== TRAINING ========================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Visualization
    writer = tf.summary.FileWriter('graph_viz/')
    writer.add_graph(tf.get_default_graph())

    se3_losses_history = []
    fc_losses_history = []

    for i_epoch in range(num_epochs):
        data_inputs = np.random.random([timesteps, batch_size, 1280, 384, 6])
        data_se3_labels = np.random.random([timesteps, batch_size, 7])
        data_fc_labels = np.random.random([timesteps, batch_size, 6])

        curr_lstm_states = np.zeros([2, 2, batch_size, 256])

        _se3_losses, _se3_trainer, _curr_lstm_states = sess.run(
            [se3_losses, se3_trainer, lstm_states, ],
            feed_dict={
                inputs: data_inputs,
                se3_labels: data_se3_labels,
                lstm_init_state: curr_lstm_states,
                se3_lr: 0.001,
            }
        )
        se3_losses_history.append(_se3_losses)
        curr_lstm_states = _curr_lstm_states

        _fc_losses, _fc_trainer, _curr_lstm_states = sess.run(
            [fc_losses, fc_trainer, lstm_states, ],
            feed_dict={
                inputs: data_inputs,
                fc_labels: data_fc_labels,
                lstm_init_state: curr_lstm_states,
                fc_lr: 0.001,
            }
        )
        fc_losses_history.append(_fc_losses)

        curr_lstm_states = _curr_lstm_states

        # print stats
        print("se_loss: %f, fc_loss: %f" % (_se3_losses, _fc_losses))
