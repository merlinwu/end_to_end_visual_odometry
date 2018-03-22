import tensorflow as tf


def se3_losses(outputs, labels, k=1):
    diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]
    diff_q = outputs[:, :, 3:] - labels[:, :, 3:]

    # takes the the dot product and sum it up along time
    sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(0, 2,))
    sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(0, 2,))

    t = tf.cast(tf.shape(outputs)[0], tf.float32)

    # multiplies the sum by 1 / t
    loss = (sum_diff_p_dot_p + k * sum_diff_q_dot_q) / t

    with tf.Session() as sess:
        print(t.eval())
        print(sum_diff_p_dot_p.eval())
        print(sum_diff_q_dot_q.eval())
        print(loss.eval())
        print(tf.reduce_mean(loss).eval())

    return tf.reduce_mean(loss)


outputs = tf.constant([
    [[1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 1, 1, 2], [2, 0, 1, 0, 0, 2, 0]],
    [[1, 1, 1, 1, 1, 1, 2], [2, 0, 1, 0, 0, 2, 0]],
], dtype=tf.float32)

labels = tf.constant([
    [[8, 9, 10, 11, 12, 13, 14], [0, 2, 0, 0, 0, 0, 0]],
    [[1, 2, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]],
    [[1, 2, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]],

], dtype=tf.float32)
se3_losses(outputs, labels, 1)
