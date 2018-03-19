import tensorflow as tf


def se3_losses(outputs, labels, k=1):
    diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]
    diff_q = outputs[:, :, 3:] - labels[:, :, 3:]

    # takes the the dot product and sum it up along time
    sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(1, 2,))
    sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(1, 2,))

    # multiplies the sum by 1 / t
    loss = (sum_diff_p_dot_p + k * sum_diff_q_dot_q) / tf.cast(tf.shape(outputs)[1], tf.float32)

    return tf.reduce_mean(loss)


def fc_losses(outputs, labels):
    pass
