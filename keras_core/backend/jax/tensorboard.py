import jax


def start_trace(logdir):
    raise NotImplementedError
    jax.profiler.start_trace(logdir)


def stop_trace(save):
    raise NotImplementedError
    # TODO: figure out jaxlib.xla_extension.XlaRuntimeError:
    # # NOT_FOUND: plugins; No such file or directory
    # always saves, `save` argument included for tf compatibility
    jax.profiler.stop_trace()
