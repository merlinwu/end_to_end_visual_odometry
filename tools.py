from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
    """foldl that returns results after each iteration instead of just the last
    one. It is the same as the tensorflow foldl otherwise
    """
    if not callable(fn):
        raise TypeError("fn must be callable.")

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

        if initializer is None:
            a = elems_ta.read(0)
            i = constant_op.constant(1)
        else:
            a = ops.convert_to_tensor(initializer)
            i = constant_op.constant(0)

        def compute(i, a):
            a = fn(a, elems_ta.read(i))
            return [i + 1, a]

        _, r_a = control_flow_ops.while_loop(
            lambda i, a: i < n, compute, [i, a],
            parallel_iterations=parallel_iterations,
            back_prop=back_prop,
            swap_memory=swap_memory)

        # TODO(akshayka): Remove the in_graph_mode check once caching devices are
        # supported in Eager
        if in_graph_mode and varscope_caching_device_was_none:
            varscope.set_caching_device(None)
        return r_a
