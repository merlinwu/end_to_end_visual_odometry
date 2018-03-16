import tensorflow as tf


def se3_losses(outputs, labels, k):
    diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]
    diff_q = outputs[:, :, 3:] - labels[:, :, 3:]

    sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(1, 2,))
    sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(1, 2,))

    loss = (sum_diff_p_dot_p + sum_diff_q_dot_q) / tf.cast(tf.shape(outputs)[0], tf.float32)

    return tf.reduce_mean(loss)


def fc_losses(outputs, labels):
    pass
