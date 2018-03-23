import tensorflow as tf


def fc_losses(outputs, labels_u):
    diff_u = outputs[:, :, 0:6] - labels_u
    L = outputs[:, :, 6:12]

    # The network outputs Q=LL* through the Cholesky decomposition,
    # we assume L is diagonal, Q is always psd
    Q = tf.multiply(L, L)

    # determinant of a diagonal matrix is product of it diagonal
    det_Q = tf.reduce_prod(Q, axis=2)

    # inverse of a diagonal matrix is elemental inverse
    inv_Q = tf.div(tf.constant(1, dtype=tf.float32), Q + 1e-8)

    # sum of determinants along the time
    sum_det_Q = tf.reduce_sum(det_Q, axis=0)

    # sum of diff_u' * inv_Q * diff_u
    s = tf.reduce_sum(tf.multiply(diff_u, tf.multiply(inv_Q, diff_u)), axis=(0, 2,))

    t = tf.cast(tf.shape(outputs)[0], tf.float32)

    # add and multiplies of sum by 1 / t
    loss = (s + sum_det_Q) / t

    with tf.Session() as sess:
        print("diff_u", diff_u.eval())
        print("L", L.eval())
        print("Q", Q.eval())
        print("det_Q", det_Q.eval())
        print("inv_Q", inv_Q.eval())
        print("sum_det_Q", sum_det_Q.eval())
        print("s", s.eval())
        print("loss", loss.eval())
        print("t", t.eval())
        print("", tf.reduce_mean(loss).eval())

    return tf.reduce_mean(loss)


outputs = tf.constant([
    [[7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, ], [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, ]],
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ]],

], dtype=tf.float32)

labels = tf.constant([
    [[2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]],
    [[7, 2, 3, 4, 5, 6], [7, 2, 3, 4, 5, 6]],

], dtype = tf.float32)
fc_losses(outputs, labels)
