import tensorflow as tf

# assumes time major
def se3_losses(outputs, labels, k):
    with tf.variable_scope("se3_losses"):
        diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]
        diff_q = outputs[:, :, 3:] - labels[:, :, 3:]

        # takes the the dot product and sum it up along time
        sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(0, 2,))
        sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(0, 2,))

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # multiplies the sum by 1 / t
        loss = (sum_diff_p_dot_p + k * sum_diff_q_dot_q) / t

        return tf.reduce_mean(loss)

# assumes time major
def fc_losses(outputs, labels_u):
    with tf.variable_scope("fc_losses"):
        diff_u = outputs[:, :, 0:6] - labels_u
        L = outputs[:, :, 6:12]

        # The network outputs Q=LL* through the Cholesky decomposition,
        # we assume L is diagonal, Q is always psd
        Q = tf.multiply(L, L)

        # determinant of a diagonal matrix is product of it diagonal
        log_det_Q = tf.log(tf.reduce_prod(Q, axis=2))

        # inverse of a diagonal matrix is elemental inverse
        inv_Q = tf.div(tf.constant(1, dtype=tf.float32), Q + 1e-8)

        # sum of determinants along the time
        sum_det_Q = tf.reduce_sum(log_det_Q, axis=0)

        # sum of diff_u' * inv_Q * diff_u
        s = tf.reduce_sum(tf.multiply(diff_u, tf.multiply(inv_Q, diff_u)), axis=(0, 2,))

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # add and multiplies of sum by 1 / t
        loss = (s + sum_det_Q) / t

        return tf.reduce_mean(loss)
