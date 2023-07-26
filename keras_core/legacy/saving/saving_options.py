import contextlib
import threading


class SaveOptionsContext(threading.local):
    def __init__(self):
        self.use_legacy_config = False


_save_options_context = SaveOptionsContext()


@contextlib.contextmanager
def keras_option_scope(use_legacy_config=True):
    use_legacy_config_prev_value = _save_options_context.use_legacy_config
    try:
        _save_options_context.use_legacy_config = use_legacy_config
        yield
    finally:
        _save_options_context.use_legacy_config = use_legacy_config_prev_value


def in_legacy_saving_scope():
    return _save_options_context.use_legacy_config
