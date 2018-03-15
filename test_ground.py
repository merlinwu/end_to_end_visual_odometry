from se3 import *

a = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
b = tf.constant([7, 8, 9, 3, 2.3, 2.2], dtype=tf.float32)
c = se3_comp(pose_ypr_to_quat(b), a)

with tf.Session() as sess:
    print(a.eval())
    print(b.eval())
    print(c.eval())