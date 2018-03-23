from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
import tensorflow as tf


# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def foldl(fn, elems, dtype=None, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
    """foldl that returns results after each iteration instead of just the last
    one. It is the same as the tensorflow foldl otherwise
    """
    if not callable(fn):
        raise TypeError("fn must be callable.")

    if initializer is None:
        raise TypeError("There must be a initializer")

    if dtype is None:
        raise TypeError("There must be a type")

    in_graph_mode = context.in_graph_mode()
    with ops.name_scope(name, "foldl", [elems]):
        # TODO(akshayka): Remove the in_graph_mode check once caching devices are
        # supported in Eager
        if in_graph_mode:
            # Any get_variable calls in fn will cache the first call locally
            # and not issue repeated network I/O requests for each iteration.
            varscope = vs.get_variable_scope()
            varscope_caching_device_was_none = False
            if varscope.caching_device is None:
                # TODO(ebrevdo): Change to using colocate_with here and in other
                # methods.
                varscope.set_caching_device(lambda op: op.device)
                varscope_caching_device_was_none = True

        # Convert elems to tensor array.
        elems = ops.convert_to_tensor(elems, name="elems")
        n = array_ops.shape(elems)[0]
        elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                                dynamic_size=False,
                                                infer_shape=True)
        elems_ta = elems_ta.unstack(elems)

        a = ops.convert_to_tensor(initializer)
        i = constant_op.constant(1)

        ta = tensor_array_ops.TensorArray(dtype=dtype, size=n + 1, dynamic_size=False, infer_shape=True,
                                          clear_after_read=False)
        ta = ta.write(0, a)

        def compute(i, ta):
            a = fn(ta.read(i - 1), elems_ta.read(i - 1))
            ta = ta.write(i, a)
            return [i + 1, ta]

        _, r_a = control_flow_ops.while_loop(
            lambda i, a: i < n + 1, compute, [i, ta],
            parallel_iterations=parallel_iterations,
            back_prop=back_prop,
            swap_memory=swap_memory)

        # TODO(akshayka): Remove the in_graph_mode check once caching devices are
        # supported in Eager
        if in_graph_mode and varscope_caching_device_was_none:
            varscope.set_caching_device(None)

        r_a = r_a.stack()
        r_a = r_a[1:]
        r_a.set_shape(elems.get_shape())
        return r_a


def static_map_fn(fn, inputs, axis):
    with tf.variable_scope("static_map_fn"):
        unstacked_inputs = tf.unstack(inputs, axis=axis)

        outputs = []

        for val in unstacked_inputs:
            outputs.append(fn(val))

        return tf.stack(outputs, axis=axis)
