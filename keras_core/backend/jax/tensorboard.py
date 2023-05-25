import jax


def start_trace(logdir):
    jax.profiler.start_trace(logdir)


def stop_trace(save):
    # always saves, `save` argument included for tf compatibility
    jax.profiler.stop_trace()
